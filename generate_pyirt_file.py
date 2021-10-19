# TODOS
import argparse
import json
import glob

# take a data set name as input 1
# take a folder of outs as input 2

# for each model:

# score preds

# set qID to be the text itself

# write outputs to jsonlines (like this):
#

#{"subject_id": "pedro",    "responses": {"q1": 1, "q2": 0, "q3": 1, "q4": 0}}
#{"subject_id": "pinguino", "responses": {"q1": 1, "q2": 1, "q3": 0, "q4": 0}}
#{"subject_id": "ken",      "responses": {"q1": 1, "q2": 1, "q3": 1, "q4": 1}}
#{"subject_id": "burt",     "responses": {"q1": 0, "q2": 0, "q3": 0, "q4": 0}}


# hs: statement
# sentiment: statement --> label
# nli: context and hypothesis --> label (sometimes list of labels)
# qa: context, question --> answer (sometimes list with len 1)

parser = argparse.ArgumentParser()
parser.add_argument('-d', help="dataset file")
parser.add_argument('-f', help="folder for preds")
parser.add_argument('-t',  help="task")
args = parser.parse_args()

# first load my data set
task_dict = {
    "hs" : {
        "text1": "statement",
        "text2": None,
        "label": "label"
    },
    "sentiment": {
        "text1": "statement",
        "text2": None,
        "label": "label"
    },
    "nli": {
        "text1" : "context",
        "text2": "hypothesis",
        "label": "label"
    },
    "qa": {
        "text1": "context",
        "text2": "question",
        "label": "answer"
    }
}

def parse_row(d, task):
    text1 = task_dict[task]["text1"]
    text2 = task_dict[task]["text2"]
    label = task_dict[task]["label"]

    t1 = d[text1]
    if text2 is not None:
        t2 = d[text2]
        t1 += " " + t2
    l = d[label]
    return t1, l

gold_dict = dict()

with open(args.d, "r") as infile:
    for line in infile:
        d = json.loads(line)
        statement, label = parse_row(d, args.t)
        gold_dict[d["uid"]] = {
            "statement": statement,
            "label": label
        }

# load and score each response in the predictions file
# get the dataset name
dname = args.d.split("/")[-1]
preds = dict()
print(dname)
for f in glob.glob(f"{args.f}/*/*/{dname}.out"):
    print(f)
    preds[f] = {
        "responses": dict()
    }
    with open(f, "r") as infile:
        for line in infile:
            d = json.loads(line)
            gold = gold_dict[d["id"]]
            text = gold["statement"]
            label = gold["label"]
            if type(label) == list:
                label = label[0].lower()
            pred = d[task_dict[args.t]["label"]].lower()
            preds[f]["responses"][text] = int(label == pred)

outf = f"{args.d}.preds.jsonlines"

with open(outf, "w") as outfile:
    for user, user_preds in preds.items():
        output = {
            "subject_id": user,
            "responses": user_preds["responses"]
        }
        json.dump(output, outfile)
        outfile.write("\n")
    
