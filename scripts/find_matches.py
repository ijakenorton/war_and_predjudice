import re


def find(stuff):
    context = stuff["context"]
    text = stuff["text"]

    split = text.split(" ")
    window_size = 5
    pride_matches = []
    war_matches = []

    with open("./1342-0.txt", "r") as file:
        pride = file.read()

    with open("./pg2600.txt", "r") as file:
        war = file.read()

    for i in range(len(split) - window_size + 1):
        window = " ".join(split[i : i + window_size])
        pattern_pride = re.compile(re.escape(window), re.IGNORECASE)
        pattern_war = re.compile(re.escape(window), re.IGNORECASE)

        pride_match = pattern_pride.search(pride)
        war_match = pattern_war.search(war)

        if pride_match:
            pride_matches.append(window)
        if war_match:
            war_matches.append(window)

    # Print matches found
    return {
        "context": context,
        "text": text,
        "pride_matches": pride_matches,
        "war_matches": war_matches,
    }
