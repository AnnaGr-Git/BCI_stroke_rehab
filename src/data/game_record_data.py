import os
import pickle
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
# import tensorflow as tf
from pylsl import StreamInlet, resolve_stream

# https://docs.google.com/document/d/e/2PACX-1vR_4DXPTh1nuiOwWKwIZN3NkGP3kRwpP4Hu6fQmy3jRAOaydOuEI1jket6V4V6PG4yIG15H1N7oFfdV/pub
from src.data.signal_processing import apply_mix, bandpass, logvar


USER = "anna"  # This is the username
BOX_MOVE = "random"  # random or model
TASK_TYPE = "arm"
CHANNELS = ["CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "FZ", "FP1", "FP2", "F7", "F3", "F4", "F8", "T3", "C3"]
SAMPLE_RATE = 125

font = cv2.FONT_HERSHEY_COMPLEX


## ------------ FUNCTIONS ------------------------------
# Create game
def create_game_settings(WIDTH, HEIGHT, SQ_SIZE):
    '''
    Create parameters and settings of objects of the game recording visualization.
            Parameters:
                    WIDTH (int): Width of the game
                    HEIGHT (int): Height of the game
                    SQ_SIZE (int): Size of the moving cube

            Returns:
                    None
    '''
    game_settings = {}
    game_settings["square_1"] = {
        "x1": int(int(WIDTH) / 4 - int(SQ_SIZE / 2)),
        "x2": int(int(WIDTH) / 4 + int(SQ_SIZE / 2)),
        "y1": int(int(HEIGHT) / 10 * 9 - int(SQ_SIZE / 2)),
        "y2": int(int(HEIGHT) / 10 * 9 + int(SQ_SIZE / 2)),
    }

    game_settings["square_2"] = {
        "x1": int(int(WIDTH) / 4 * 3 - int(SQ_SIZE / 2)),
        "x2": int(int(WIDTH) / 4 * 3 + int(SQ_SIZE / 2)),
        "y1": int(int(HEIGHT) / 10 * 9 - int(SQ_SIZE / 2)),
        "y2": int(int(HEIGHT) / 10 * 9 + int(SQ_SIZE / 2)),
    }

    color_box_1 = [16 / 255, 128 / 255, 255 / 255]  # dark blue
    color_box_2 = [255 / 255, 95 / 255, 1 / 255]  # dark orange
    game_settings["box_1"] = np.ones(
        (
            game_settings["square_1"]["y2"] - game_settings["square_1"]["y1"],
            game_settings["square_1"]["x2"] - game_settings["square_1"]["x1"],
            3,
        )
    ) * np.array(color_box_1)
    game_settings["box_2"] = np.ones(
        (
            game_settings["square_2"]["y2"] - game_settings["square_2"]["y1"],
            game_settings["square_2"]["x2"] - game_settings["square_2"]["x1"],
            3,
        )
    ) * np.array(color_box_2)

    color_line_1 = [144 / 255, 196 / 255, 255 / 255]  # bright blue
    color_line_2 = [255 / 255, 177 / 255, 131 / 255]  # bright orange

    game_settings["vertical_line"] = np.ones((HEIGHT, 10, 3)) * np.array(color_line_1)
    game_settings["vertical_line_2"] = np.ones((HEIGHT, 10, 3)) * np.array(color_line_2)
    # game_settings['horizontal_line'] = np.ones((10, WIDTH, 3)) * np.random.uniform(size=(3,))

    if TASK_TYPE == "arm":
        img = (cv2.imread("C:/Users/annag/OneDrive - Danmarks Tekniske Universitet/Semester_04/Special_Course_BCI/03_code/BCI_stroke_rehab/data/external/arm.png")/ 255)
        # Downscale image
        scale_percent = 40  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        game_settings["img_task"] = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    else:
        game_settings["img_task"] = np.zeros((3,3,3))

    game_settings["WIDTH"] = WIDTH
    game_settings["HEIGHT"] = HEIGHT
    game_settings["SQ_SIZE"] = SQ_SIZE

    return game_settings


def environment(game_settings):
    """ Create image of game environemt based on settings """

    env = np.zeros((game_settings["HEIGHT"], game_settings["WIDTH"], 3))
    env[
        :, game_settings["WIDTH"] // 4 - 5 : game_settings["WIDTH"] // 4 + 5, :
    ] = game_settings["vertical_line"]
    env[
        :, game_settings["WIDTH"] // 4 * 3 - 5 : game_settings["WIDTH"] // 4 * 3 + 5, :
    ] = game_settings["vertical_line_2"]
    env[
        game_settings["square_1"]["y1"] : game_settings["square_1"]["y2"],
        game_settings["square_1"]["x1"] : game_settings["square_1"]["x2"],
        :,
    ] = game_settings["box_1"]
    env[
        game_settings["square_2"]["y1"] : game_settings["square_2"]["y2"],
        game_settings["square_2"]["x1"] : game_settings["square_2"]["x2"],
        :,
    ] = game_settings["box_2"]
    return env


def update(pos, game_settings, actions):
    """ Update game with the next task."""
    env = environment(game_settings)

    # Add arm image
    shape = game_settings["img_task"].shape
    start_x = int(game_settings["WIDTH"] / 2 - 70)
    start_y = int(game_settings["HEIGHT"] // 4 + 150)
    env[start_y : start_y + shape[0], start_x : start_x + shape[1], :] = game_settings[
        "img_task"
    ]

    if actions[pos] == "right":
        color = (255 / 255, 95 / 255, 1 / 255)
    elif actions[pos] == "left":
        color = (16 / 255, 128 / 255, 255 / 255)
    else:
        color = (255, 255, 255)
    cv2.putText(
        env,
        "Task " + str(pos) + ":",
        (int(game_settings["WIDTH"] / 2 - 70), int(game_settings["HEIGHT"] // 4)),
        font,
        1,
        (255, 255, 255),
        2,
    )  # text,coordinate,font,size of text,color,thickness of font
    cv2.putText(
        env,
        str(actions[pos]),
        (int(game_settings["WIDTH"] / 2 - 70), int(game_settings["HEIGHT"] // 4 + 70)),
        font,
        2,
        color,
        3,
    )  # text,coordinate,font,size of text,color,thickness of font

    cv2.imshow("", env)
    cv2.waitKey(500)


def move(game_settings):
    """ Move cube based on previous task and show relax-command"""

    env = environment(game_settings)
    cv2.putText(
        env,
        "Relax ...",
        (int(game_settings["WIDTH"] / 2 - 70), int(game_settings["HEIGHT"] // 4)),
        font,
        1,
        (255, 255, 255),
        2,
    )  # text,coordinate,font,size of text,color,thickness of font
    cv2.imshow("", env)
    cv2.waitKey(1000)

def game_startscreen(game_settings):
    """ Show inital screen of game before recording """
    env = environment(game_settings)
    cv2.putText(
        env,
        "To start the game:",
        (int(game_settings["WIDTH"] / 3-20), int(game_settings["HEIGHT"] // 4)),
        font,
        1,
        (255, 255, 255),
        2,
    )  # text,coordinate,font,size of text,color,thickness of font
    cv2.putText(
        env,
        "Press ENTER ...",
        (int(game_settings["WIDTH"] / 3-20), int(game_settings["HEIGHT"] // 4 + 70)),
        font,
        1,
        (255, 255, 255),
        2,
    )  # text,coordinate,font,size of text,color,thickness of font
    cv2.imshow("", env)
    cv2.waitKey(0)


def generate_random_actions(n: int = 20):
    """ Generate random sequence of n actions per side."""
    # Action to be performed
    actions = np.full(n, "right")
    actions = np.append(actions, np.full(n, "left"))
    np.random.shuffle(actions)
    return actions


def input_process(trial, W_matrix):
    trial = np.expand_dims(trial, axis=2)
    n_channels = trial.shape[0]
    nsamples_win = trial.shape[1]
    sample_rate = 125

    filter_input = bandpass(trial, 8, 15, sample_rate, n_channels, nsamples_win)
    output_mix = apply_mix(W_matrix, filter_input, n_channels, nsamples_win)
    comp = np.array([0, -2])
    output_mix = output_mix[comp, :, 0]
    point = logvar(output_mix)

    return point


def predict_nn(input_nn, model, W_matrix):
    input_nn = input_process(input_nn, W_matrix)
    out = model.predict(input_nn.reshape(1, 2))
    print(out)
    output = np.rint(out)
    return output


def predict_lda(input_lda, c, d, W_matrix):
    print("Predicting")
    point = input_process(input_lda, W_matrix)
    results = np.cross(point - d, c - d)
    print(point, results)
    # 1 left 0 right
    if results < 0:
        # above the line
        output = 0
    else:
        output = 1

    return output


## ------------ MAIN ------------------------------
def main(n_samples: int = 20):
    # Get game settings
    WIDTH = 800
    HEIGHT = 600
    SQ_SIZE = 50
    MOVE_SPEED = 1
    game_settings = create_game_settings(WIDTH, HEIGHT, SQ_SIZE)

    # Where data will be stored
    date = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    data_dir = "data/" + "raw/" + USER + "/" + date
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Load model if applied
    if BOX_MOVE == "model":
        MODEL_NAME = "models_07/NN/64.15-50epoch-1657286289-loss-0.64.model "
        # model = tf.keras.models.load_model(MODEL_NAME)
        model = []

        with open(
            "/home/nurife/BCI/IM/BCI/models_09/W_matrix/1657359658.pickle", "rb"
        ) as file:
            W_load = pickle.load(file)
        W_matrix = W_load["W_matrix"]

        with open(
            "/home/nurife/BCI/IM/BCI/models_09/LDA/50.0_r_62.5_acc_1657359658.pickle",
            "rb",
        ) as file:
            LDA_model = pickle.load(file)
        W = LDA_model["W"]
        b = LDA_model["b"]

        x = np.array([-5, 1])
        y = (b - W[0] * x) / W[1]
        d = np.array([x[0], y[0]])
        c = np.array([x[1], y[1]])
    else:
        MODEL_NAME = ""

    # Get datastream: resolve an EEG stream on the lab network
    print("Looking for an EEG stream...")
    streams = resolve_stream("type", "EEG")
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0], max_buflen=4096)

    # Get random sequence of actions
    actions = generate_random_actions(n_samples)

    # Show inital screen of game for 20 s
    #game_startscreen(game_settings)
    time.sleep(1)

    # Start recording
    total = 0
    left = 0
    right = 0
    correct = 0
    channel_datas = []
    start = time.time()
    for j in range(len(actions)):
        # Update Game with new task
        #update(j, game_settings, actions)
        channel_data = {}
        time_stamps = [0]
        first_timestamp = 0
        print("Start recording Data")

        # Record for 5 seconds
        timeStampsIn5Sec = [];

        print("starting with pull chunk")
        sample_count = 0;
        samples_1, timeStamps_1 = inlet.pull_chunk();
        while True:
            time.sleep(0.1)
            samples, timeStamps = inlet.pull_chunk();            
            if len(samples) > 0:
                sample_count += len(samples);
                
                timeStampsIn5Sec.extend(timeStamps)

                elapsedTime = timeStampsIn5Sec[-1] - timeStampsIn5Sec[0];
                

                print(sample_count, "   ", len(samples), "    ", elapsedTime);

                if elapsedTime > 5:
                    break;

        print("stopepd with pull chunk")

        
        while time_stamps[-1] < 5:
            # Get sample fomr EEG stream
            sample, timestamp = inlet.pull_sample()
            
        
            timestamp = time.time()
            if timestamp > first_timestamp + time_stamps[-1]:
                for i in range(16):
                    if i not in channel_data:
                        channel_data[i] = [sample[i]]
                    else:
                        channel_data[i].append(sample[i])
                if first_timestamp == 0:
                    first_timestamp = timestamp
                time_stamps.append(timestamp - first_timestamp)
                # print(timestamp - first_timestamp)



        print("End recording Data")

        # Transform data in df
        channels = {}
        channels["class"] = str(TASK_TYPE)+"_"+str(actions[j])
        channels["time_in_s"] = time_stamps[1:]
        for i in range(len(channel_data)):
            channels[CHANNELS[i]] = channel_data[i]
        channels = pd.DataFrame(channels)

        # Move box in video and show relax command
        if BOX_MOVE == "random":
            box = actions[j]
            box_shift = int((0.8 * game_settings["HEIGHT"]) // n_samples)
            if box == "left":
                game_settings["square_1"]["y1"] -= box_shift
                game_settings["square_1"]["y2"] -= box_shift
            elif box == "right":
                game_settings["square_2"]["y1"] -= box_shift
                game_settings["square_2"]["y2"] -= box_shift

        elif BOX_MOVE == "model":
            choice = predict_lda(
                np.asarray(channels)[:460, 2:].astype("float32").transpose(),
                c,
                d,
                W_matrix,
            )

            if choice == 1:
                print(f"Action:{actions[j]} Moving:left")
                game_settings["square_1"]["y1"] -= 20
                game_settings["square_1"]["y2"] -= 20
                left += 1
            elif choice == 0:
                print(f"Action:{actions[j]} Moving:right")
                game_settings["square_2"]["y1"] -= 20
                game_settings["square_2"]["y2"] -= 20
                right += 1

        #move(game_settings)
        time.sleep(1)
        total += 1
        curr_time = int(time.time())
        save_path = os.path.join(f"{data_dir}/", f"{TASK_TYPE}_{actions[j]}_{j}_{curr_time}.csv")
        channels.to_csv(save_path)

        channel_datas.append(channels)

    # plt.plot(channel_datas[0][0])
    # plt.show()

    print(time.time() - start)
    print(len(channel_datas))
    print("Done.")

    print(USER, correct / total)
    print(f"left: {left / total}, right: {right / total}")

    with open("accuracies.csv", "a") as f:
        f.write(
            f"{int(time.time())},{USER},{correct / total},{MODEL_NAME},{left / total},{right / total}\n"
        )


if __name__ == "__main__":
    main()
