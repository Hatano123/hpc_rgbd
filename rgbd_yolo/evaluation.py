"""Persist evaluation numbers next to the model and validation split."""
import json
from pathlib import Path


def save_evaluation(metrics, output, model, data, split):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    classes = []
    for i, class_id in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        classes.append({'class_id': int(class_id), 'name': metrics.names[int(class_id)],
                        'precision': float(p), 'recall': float(r), 'AP50': float(ap50), 'AP50_95': float(ap)})
    result = {'model': str(Path(model).resolve()), 'data': str(Path(data).resolve()),
              'split': split,
              'metrics': {key: float(value) for key, value in metrics.results_dict.items()},
              'classes_with_ground_truth': classes, 'speed_ms_per_image': metrics.speed,
              'ultralytics_output': str(metrics.save_dir)}
    (output / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Evaluation', '', f'- Split: {split}', '',
             '| Class with ground truth | Precision | Recall | AP50 | AP50-95 |', '|---|---:|---:|---:|---:|']
    lines += [f"| {item['name']} | {item['precision']:.4f} | {item['recall']:.4f} | {item['AP50']:.4f} | {item['AP50_95']:.4f} |" for item in classes]
    lines += ['', 'Classes absent from this split are not evaluated; a small smoke test does not measure production accuracy.']
    (output / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result
