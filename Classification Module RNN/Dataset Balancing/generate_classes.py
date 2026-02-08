import os, json
base='d:/MAIN PROJECT/UnderWater-Acoustic-Detection/data_balanced'
classes=[]
for cat in ['ships','marine_life']:
    cat_path=os.path.join(base,cat)
    if not os.path.exists(cat_path):
        continue
    for sub in sorted(os.listdir(cat_path)):
        if os.path.isdir(os.path.join(cat_path,sub)):
            classes.append(f"{cat}_{sub}")
with open('d:/MAIN PROJECT/UnderWater-Acoustic-Detection/classes.json','w') as f:
    json.dump(classes,f)
print('Wrote classes.json with',len(classes),'classes')
