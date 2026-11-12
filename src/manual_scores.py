"""Recorded AI-generated proxy review (Claude), coherence/responsiveness 1-5.

Claude assigned these scores; they are not independent human review.
They have not been independently judge-validated; no judge API cross-check ran.
The fixed sample spans 9 prompts per config. This script only re-exports the
recorded scores; it performs no new review. Legacy manual_* filenames are kept
for compatibility. Rubric: lipogram spellings and factual errors not penalised.
"""
import csv, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAMPLE = ["p01","p06","p13","p22","p25","p29","p33","p37","p40"]
# (coherence, responsiveness)
SCORES = {
  "base_mask": {"p01":(2,3),"p06":(3,4),"p13":(4,5),"p22":(4,4),"p25":(2,2),
                "p29":(1,2),"p33":(5,5),"p37":(5,5),"p40":(4,4)},
  "adapter_mask": {"p01":(5,5),"p06":(5,5),"p13":(5,5),"p22":(5,5),"p25":(5,5),
                   "p29":(5,5),"p33":(4,5),"p37":(5,5),"p40":(5,5)},
}

def main():
    rows=[]
    agg={}
    for cfg, sc in SCORES.items():
        scored={json.loads(l)["id"]:json.loads(l) for l in open(f"{ROOT}/results/scored/{cfg}.jsonl")}
        coh=[]; resp=[]
        for pid in SAMPLE:
            r=scored[pid]; c,rp=sc[pid]; coh.append(c); resp.append(rp)
            rows.append([pid,cfg,r["req"]["type"],r["answer"].replace("\n"," "),
                         r["scores"]["letter_compliant"],r["scores"]["looks_english"],
                         r["scores"]["requirement"]["satisfied"],c,rp,"AI-generated proxy review (Claude); not independent human review; not independently judge-validated"])
        agg[cfg]={"review_source":"AI-generated proxy review (Claude)","independent_human_review":False,"independent_judge_validation":False,"n":len(coh),"mean_coherence":round(sum(coh)/len(coh),2),
                  "mean_responsiveness":round(sum(resp)/len(resp),2)}
    with open(f"{ROOT}/results/manual_review_scored.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["id","config","req_type","answer","letter_ok","english_ok","req_ok","coherence_1to5","responsiveness_1to5","review_source"])
        w.writerows(rows)
    json.dump(agg, open(f"{ROOT}/results/manual_summary.json","w"), indent=2)
    print(json.dumps(agg, indent=2))

if __name__=="__main__":
    main()
