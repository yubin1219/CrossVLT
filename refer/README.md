## Note
This API is able to load all 3 referring expression datasets, i.e., RefCOCO, RefCOCO+ and RefCOCOg. 
They are with different train/val/test split by UNC, Google and UMD respectively. We provide all kinds of splits here.


## Setup
Run "make" before using the code.
It will generate ``_mask.c`` and ``_mask.so`` in ``external/`` folder.
These mask-related codes are copied from mscoco [API](https://github.com/pdollar/coco).

## Download
Download the cleaned data and extract them into "data" folder
- 1) http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
- 2) http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip 
- 3) http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip 

## Prepare Images:
Besides, add "mscoco" into the ``data/images`` folder, which can be from [mscoco](http://mscoco.org/dataset/#overview)
COCO's images are used for RefCOCO, RefCOCO+ and refCOCOg.

## How to use
The "refer.py" is able to load all 3 datasets with different kinds of data split by UNC and UMD.
**Note for RefCOCOg, we suggest use UMD's split which has train/val/test splits and there is no overlap of images between different split.**
```bash
# locate your own data_root, and choose the dataset_splitBy you want to use
refer = REFER(data_root, dataset='refcoco',  splitBy='unc')
refer = REFER(data_root, dataset='refcoco+', splitBy='unc')
refer = REFER(data_root, dataset='refcocog', splitBy='umd')      # Recommended, including train/val/test
```


<!-- refs(dataset).p contains list of refs, where each ref is
{ref_id, ann_id, category_id, file_name, image_id, sent_ids, sentences}
ignore filename

Each sentences is a list of sent
{arw, sent, sent_id, tokens}
 -->
