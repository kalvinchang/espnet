from tqdm import tqdm

task = "test"

f = open(f"data/{task}_text.ctc", "r")
g = open(f"/ocean/projects/cis210027p/eyeo1/workspace/espnet/egs2/ipapack/asr1/local/data2/{task}_text", "w")

for line in tqdm(f.readlines()):
    line = line.strip()
    utt_id, text = line.split(" ", 1)
    text = text.replace("//", " ").replace("/","")
    # print(f"{utt_id} {text}")
    g.write(f"{utt_id} {text}\n")

f.close()
g.close()