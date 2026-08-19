from flask import Flask, render_template, request, redirect, url_for, session
import tensorflow as tf
import tf_keras as keras
import numpy as np
from PIL import Image
import os


app = Flask(__name__)

app.secret_key = "rps-secret-key-123"


# =====================================================
# MODEL
# =====================================================

MODEL_PATH = "keras_model.h5"
LABEL_PATH = "labels.txt"


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("keras_model.h5 not found!")

if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError("labels.txt not found!")


print("Loading model...")

model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded!")


# =====================================================
# LABELS
# =====================================================

labels = []

canonical_labels = {
    "rock": "Rock",
    "paper": "Paper",
    "scissor": "Scissor",
    "scissors": "Scissor"
}

with open(LABEL_PATH, "r") as file:

    for line in file:

        line = line.strip()

        if line:

            parts = line.split(" ", 1)

            if len(parts) == 2:
                raw_label = parts[1].strip()
            else:
                raw_label = parts[0].strip()

            normalized_label = canonical_labels.get(
                raw_label.lower()
            )

            if normalized_label is None:
                raise ValueError(
                    f"Unsupported model label: {raw_label}"
                )

            labels.append(normalized_label)


print("Labels:", labels)

if len(labels) != 3 or set(labels) != {
    "Rock",
    "Paper",
    "Scissor"
}:
    raise ValueError(
        "labels.txt must contain Rock, Paper, and Scissor in model output order."
    )


# =====================================================
# MODEL INPUT SIZE
# =====================================================

input_shape = model.input_shape

if isinstance(input_shape, list):
    input_shape = input_shape[0]

MODEL_HEIGHT = input_shape[1]
MODEL_WIDTH = input_shape[2]
MIN_CONFIDENCE = 50


# =====================================================
# PREDICT
# =====================================================

def predict_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (MODEL_WIDTH, MODEL_HEIGHT)
    )

    image_array = np.asarray(
        image
    ).astype(np.float32)

    image_array = (
        image_array / 127.5
    ) - 1

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )

    index = int(
        np.argmax(prediction[0])
    )

    confidence = (
        float(prediction[0][index])
        * 100
    )

    move = labels[index]

    return move, confidence


# =====================================================
# WINNER
# =====================================================

def get_winner(player1, player2):

    if player1 == player2:
        return "DRAW"

    if player1 == "Rock" and player2 == "Scissor":
        return "PLAYER 1"

    if player1 == "Scissor" and player2 == "Paper":
        return "PLAYER 1"

    if player1 == "Paper" and player2 == "Rock":
        return "PLAYER 1"

    return "PLAYER 2"


# =====================================================
# HOME
# =====================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =====================================================
# START GAME
# =====================================================

@app.route(
    "/start",
    methods=["POST"]
)
def start():

    rounds = request.form.get(
        "rounds",
        "5"
    )

    rounds = int(rounds)

    session.clear()

    session["total_rounds"] = rounds

    session["current_round"] = 1

    session["score1"] = 0

    session["score2"] = 0

    session["player1_move"] = None

    session["history"] = []

    return redirect(
        url_for("game")
    )


# =====================================================
# GAME PAGE
# =====================================================

@app.route("/game")
def game():

    if "total_rounds" not in session:

        return redirect(
            url_for("index")
        )

    return render_template(
        "game.html",

        total_rounds=
            session["total_rounds"],

        current_round=
            session["current_round"],

        score1=
            session["score1"],

        score2=
            session["score2"],

        player1_move=
            session.get(
                "player1_move"
            ),

        history=
            session.get(
                "history",
                []
            )
    )


# =====================================================
# PLAYER 1
# =====================================================

@app.route(
    "/player1",
    methods=["POST"]
)
def player1():

    if "image" not in request.files:

        return redirect(
            url_for("game")
        )

    image_file = request.files["image"]

    if image_file.filename == "":

        return redirect(
            url_for("game")
        )

    try:

        image = Image.open(
            image_file
        )

        move, confidence = \
            predict_image(image)

        if confidence < MIN_CONFIDENCE:

            return render_template(
                "game.html",

                total_rounds=
                    session["total_rounds"],

                current_round=
                    session["current_round"],

                score1=
                    session["score1"],

                score2=
                    session["score2"],

                player1_move=None,

                history=
                    session.get(
                        "history",
                        []
                    ),

                error=
                    "Hand not detected clearly. Please try again."
            )


        # Store Player 1 move
        # BUT DON'T SHOW IT

        session["player1_move"] = move

        session.modified = True

        return redirect(
            url_for("game")
        )

    except Exception as error:

        print(error)

        return redirect(
            url_for("game")
        )


# =====================================================
# PLAYER 2
# =====================================================

@app.route(
    "/player2",
    methods=["POST"]
)
def player2():

    if "image" not in request.files:

        return redirect(
            url_for("game")
        )

    image_file = request.files["image"]

    if image_file.filename == "":

        return redirect(
            url_for("game")
        )

    player1_move = session.get(
        "player1_move"
    )

    if player1_move is None:

        return redirect(
            url_for("game")
        )

    try:

        image = Image.open(
            image_file
        )

        player2_move, confidence = \
            predict_image(image)

        if confidence < MIN_CONFIDENCE:

            return render_template(
                "game.html",

                total_rounds=
                    session["total_rounds"],

                current_round=
                    session["current_round"],

                score1=
                    session["score1"],

                score2=
                    session["score2"],

                player1_move=
                    player1_move,

                history=
                    session.get(
                        "history",
                        []
                    ),

                error=
                    "Hand not detected clearly. Please try again."
            )


        # Calculate winner

        winner = get_winner(
            player1_move,
            player2_move
        )


        # Update score

        score1 = session["score1"]

        score2 = session["score2"]


        if winner == "PLAYER 1":

            score1 += 1

        elif winner == "PLAYER 2":

            score2 += 1


        session["score1"] = score1

        session["score2"] = score2


        # Save history

        history = session.get(
            "history",
            []
        )


        history.append({

            "round":
                session["current_round"],

            "player1":
                player1_move,

            "player2":
                player2_move,

            "winner":
                winner

        })


        session["history"] = history

        session.modified = True


        # Store result temporarily

        session["last_player1"] = \
            player1_move

        session["last_player2"] = \
            player2_move

        session["last_winner"] = \
            winner


        return render_template(

            "game.html",

            total_rounds=
                session["total_rounds"],

            current_round=
                session["current_round"],

            score1=
                score1,

            score2=
                score2,

            player1_move=
                player1_move,

            player2_move=
                player2_move,

            winner=
                winner,

            history=
                history,

            result=True

        )


    except Exception as error:

        print(error)

        return redirect(
            url_for("game")
        )


# =====================================================
# NEXT ROUND
# =====================================================

@app.route("/next")
def next_round():

    total_rounds = \
        session["total_rounds"]

    current_round = \
        session["current_round"]


    if current_round >= total_rounds:

        return redirect(
            url_for("final")
        )


    session["current_round"] = \
        current_round + 1

    session["player1_move"] = None

    session.pop(
        "last_player1",
        None
    )

    session.pop(
        "last_player2",
        None
    )

    session.pop(
        "last_winner",
        None
    )

    session.modified = True


    return redirect(
        url_for("game")
    )


# =====================================================
# FINAL RESULT
# =====================================================

@app.route("/final")
def final():

    score1 = session.get(
        "score1",
        0
    )

    score2 = session.get(
        "score2",
        0
    )

    total_rounds = \
        session.get(
            "total_rounds",
            5
        )


    if score1 > score2:

        winner = "PLAYER 1"

    elif score2 > score1:

        winner = "PLAYER 2"

    else:

        winner = "DRAW"


    return render_template(

        "final.html",

        score1=score1,

        score2=score2,

        total_rounds=
            total_rounds,

        winner=winner

    )


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )