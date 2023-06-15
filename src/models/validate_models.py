import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection


from src.data.make_dataset import BCIDataset
from src.data.signal_processing import logvar
from src.models.ML_models import NN_model,LDA_model,QDA_model,XGB_model,SVM_model


def validate_models_CV(data_root, subjects, measurements, num_folds:int = 5, num_components:list=[2]):
    # Define Trainingset
    trainingset = BCIDataset(data_root, subjects, measurements, measurement_length=3)

    # Validate data
    trainingset.validate_data()
    # Bandpass filtering
    trainingset.apply_bandpass_filtering(selected_data="sample")

    # Split data with Stratified K-Fold for validation
    y_labels = np.array(trainingset.data['class'])
    skf = model_selection.StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=38)
    accuracies = {}
    for c in num_components:
        accuracies[str(c)+"_components"] = {'LDA': {'train':[],'test':[]},
                              'QDA': {'train':[],'test':[]},
                              'XGB': {'train':[],'test':[]},
                              'SVM': {'train':[],'test':[]},
                              'NN': {'train':[],'test':[]}}


    for i, (train_index, test_index) in enumerate(skf.split(np.zeros(len(y_labels)), y_labels)):
        print(f"Fold {i}")
        print(f"Train percentage: {len(train_index)/(len(train_index)+len(test_index))}")
        ## Get data
        # Create column of train/test-flag
        train_test_list = []
        for i in range(len(trainingset.data)):
            if i in train_index:
                train_test_list.append("train")
            elif i in test_index:
                train_test_list.append("test")
            else:
                print("Index is not used in training.")
                
        trainingset.data['train_split'] = train_test_list

        # Get CSP-features
        for num_c in num_components:
            data_comp = trainingset.feature_extraction_CSP(only_train=True,selected_data='filtered', num_components=num_c)
            # Calc logvar of features
            features = logvar(data_comp)
            print(f"Shape features: {np.shape(features)}")
            # Get train and test features
            X_train = features[train_index]
            X_test = features[test_index]
            # Get labels
            y_train = np.array(y_labels[train_index])
            y_test = np.array(y_labels[test_index])
            # Transform labels in int
            classes = np.unique(y_train)
            y_train_int = []
            for y in y_train:
                for cl_idx in range(len(classes)):
                    if y == classes[cl_idx]:
                        y_train_int.append(cl_idx)
            y_test_int = []
            for y in y_test:
                for cl_idx in range(len(classes)):
                    if y == classes[cl_idx]:
                        y_test_int.append(cl_idx)

            y_train = np.array(y_train_int)
            y_test = np.array(y_test_int)

            ## Validate Classification methods
            # LDA
            clf_lda = LDA_model()
            clf_lda = clf_lda.fit(X_train, y_train)
            accuracies[str(num_c)+"_components"]['LDA']['train'].append(accuracy_score(y_train, clf_lda.predict(X_train)))
            accuracies[str(num_c)+"_components"]['LDA']['test'].append(accuracy_score(y_test, clf_lda.predict(X_test)))

            # QDA
            clf_qda = QDA_model()
            clf_qda.fit(X_train, y_train)
            accuracies[str(num_c)+"_components"]['QDA']['train'].append(accuracy_score(y_train, clf_qda.predict(X_train)))
            accuracies[str(num_c)+"_components"]['QDA']['test'].append(accuracy_score(y_test, clf_qda.predict(X_test)))

            # XGB
            clf_xgb = XGB_model()
            clf_xgb.fit(X_train, y_train)
            accuracies[str(num_c)+"_components"]['XGB']['train'].append(accuracy_score(y_train, clf_xgb.predict(X_train)))
            accuracies[str(num_c)+"_components"]['XGB']['test'].append(accuracy_score(y_test, clf_xgb.predict(X_test)))

            # SVM
            clf_svm = SVM_model()
            clf_svm.fit(X_train, y_train)
            accuracies[str(num_c)+"_components"]['SVM']['train'].append(accuracy_score(y_train, clf_svm.predict(X_train)))
            accuracies[str(num_c)+"_components"]['SVM']['test'].append(accuracy_score(y_test, clf_svm.predict(X_test)))

            # NN
            # Split into Validation and Test set
            val_X, test_X, val_y, test_y = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)
            model = NN_model(X_train)
            # Train model
            #epochs = 80
            #batch_size = 5
            #validation_batch_size = 5

            epochs = 80
            batch_size = 10
            validation_batch_size = 5

            # epochs = 80
            # batch_size = 3
            # validation_batch_size = 1

            history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                shuffle=True,
                                validation_data=(val_X, val_y), validation_batch_size=validation_batch_size, verbose=0)
            pred = model.predict(X_train)
            pred_classes = np.rint(pred)
            accuracies[str(num_c)+"_components"]['NN']['train'].append(accuracy_score(y_train, pred_classes))
            pred = model.predict(test_X)
            pred_classes = np.rint(pred)
            accuracies[str(num_c)+"_components"]['NN']['test'].append(accuracy_score(test_y, pred_classes))

    return accuracies


def validate_models_LOSO(data_root, subjects, measurements, num_components:list=[2]):
    # Define Trainingset
    trainingset = BCIDataset(data_root, subjects, measurements, measurement_length=3)

    # Validate data
    trainingset.validate_data()
    # Bandpass filtering
    trainingset.apply_bandpass_filtering(selected_data="sample")

    accuracies = {}
    for c in num_components:
        accuracies[str(c)+"_components"] = {'SVM': {'train':[],'test':[]},
                                            'NN': {'train':[],'test':[]}}
    y_labels = np.array(trainingset.data['class'])

    # Split data into #num_subjects folds
    print(F"Subjects: {subjects}")
    for i in range(len(subjects)):
        all_indexes = list(range(len(trainingset.data)))
        # Get train and test indexes -> data of one subject=testing
        test_index = list(np.where(trainingset.data["subject"] == subjects[i])[0])
        print(f"Testing on subject {subjects[i]}.")
        train_index = list(set(all_indexes) ^ set(test_index))

        # Create column of train/test-flag
        train_test_list = []
        for i in range(len(trainingset.data)):
            if i in train_index:
                train_test_list.append("train")
            elif i in test_index:
                train_test_list.append("test")
            else:
                print("Index is not used in training.")
                
        trainingset.data['train_split'] = train_test_list

        # Get CSP-features
        for num_c in num_components:
            data_comp = trainingset.feature_extraction_CSP(only_train=True, selected_data='filtered', num_components=num_c)
            # Calc logvar of features
            features = logvar(data_comp)
            print(f"Shape features: {np.shape(features)}")
            # Get train and test features
            X_train = features[train_index]
            X_test = features[test_index]
            # Get labels
            y_train = np.array(y_labels[train_index])
            y_test = np.array(y_labels[test_index])
            # Transform labels in int
            classes = np.unique(y_train)
            y_train_int = []
            for y in y_train:
                for cl_idx in range(len(classes)):
                    if y == classes[cl_idx]:
                        y_train_int.append(cl_idx)
            y_test_int = []
            for y in y_test:
                for cl_idx in range(len(classes)):
                    if y == classes[cl_idx]:
                        y_test_int.append(cl_idx)

            y_train = np.array(y_train_int)
            y_test = np.array(y_test_int)

            ## Validate Classification methods
             # SVM
            clf_svm = SVM_model()
            clf_svm.fit(X_train, y_train)
            accuracies[str(num_c)+"_components"]['SVM']['train'].append(accuracy_score(y_train, clf_svm.predict(X_train)))
            accuracies[str(num_c)+"_components"]['SVM']['test'].append(accuracy_score(y_test, clf_svm.predict(X_test)))

            # NN
            # Split into Validation and Test set
            val_X, test_X, val_y, test_y = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)
            model = NN_model(X_train)

            # Train model
            #epochs = 80
            #batch_size = 5
            #validation_batch_size = 5

            epochs = 80
            batch_size = 10
            validation_batch_size = 5

            # epochs = 80
            # batch_size = 3
            # validation_batch_size = 1

            history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                shuffle=True,
                                validation_data=(val_X, val_y), validation_batch_size=validation_batch_size, verbose=0)
            pred = model.predict(X_train)
            pred_classes = np.rint(pred)
            accuracies[str(num_c)+"_components"]['NN']['train'].append(accuracy_score(y_train, pred_classes))
            pred = model.predict(test_X)
            pred_classes = np.rint(pred)
            accuracies[str(num_c)+"_components"]['NN']['test'].append(accuracy_score(test_y, pred_classes))
    
    return accuracies











