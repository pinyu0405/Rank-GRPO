import pickle

with open("gt_catalog.pkl", "rb") as f:
    gt_catalog = pickle.load(f)
# type(gt_catalog)
# print(type(gt_catalog))
# print(len(gt_catalog))
with open("reddit_catalog/gt_catalog.pkl", "rb") as f:
    reddit_gt_catalog = pickle.load(f)
# print(type(reddit_gt_catalog))
# print(len(reddit_gt_catalog))
# print(gt_catalog)
# compare two catalog
if gt_catalog == reddit_gt_catalog:
    print("The two catalogs are the same.")
else:
    print("The two catalogs are different.")

# 差在哪
for tuple in gt_catalog:
    if tuple not in reddit_gt_catalog:
        print(f"Tuple {tuple} is missing in reddit_gt_catalog.")


