import os
import PIL.Image
import PIL.ImageOps
from collections import namedtuple
import glob

# pred_type='best' will only return the highest confidence prediction in case multiple occurrences of labelcut are detected
# pred_type='all' will return all prediction in case multiple occurrences of labelcut are detected
def run_detections(model, image_filename, image_in_folder, model_train_image_size,
                           image_out_folder, pred_type='max', save_images=False):

    imgs = [f for f in glob.glob(image_in_folder+'*.jpg')]
    imgs.extend([f for f in glob.glob(image_in_folder+'*.png')])
    results_all = model(imgs, size=model_train_image_size)

    labelcut_list = []
    xdims = namedtuple("xdims", "xmin xmax")
    ydims = namedtuple("ydims", "ymin ymax")

    message = 'No Labelcut detected!!'
    for res in results_all.pandas().xyxy:
        if res.shape[0] > 0:
            message = 'Labelcut detected!'
            for row in res.itertuples():
                labelcut_list.append({'name': row.name,
                                      'confidence': row.confidence,
                                      'coordinates': [xdims(xmin=row.xmin, xmax=row.xmax), ydims(ymin=row.ymin, ymax=row.ymax)]})

    # sorting the predicted occurrences of labelcut by confidence score in descending order
    labelcut_list.sort(key=lambda x: x['confidence'], reverse=True)

    if message == 'Labelcut detected!':
        if save_images:
            filename_noext, _ = os.path.splitext(image_filename)
            results_all.save(os.path.join(image_out_folder, filename_noext))

        if pred_type == 'max':
            return {"detected": True, "predictions": labelcut_list[0]}
        elif pred_type == 'all':
            return {"detected": True, "predictions": labelcut_list}
        else:
            print("invalid pred_type='{}' set. expected values = ['max', 'all']". format(pred_type))
            return None

    else:
        return {"detected": False, "predictions": None}