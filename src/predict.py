import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from model import GalaxyZooModel
from dataset import GalaxyDataset, eval_transform

def generate_submission(model_weights_path, test_csv, test_img_dir, output_file='submission.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GalaxyZooModel().to(device)
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    sample_sub = pd.read_csv(test_csv)
    
    inference_dataset = GalaxyDataset(
        df=sample_sub,
        img_dir=Path(test_img_dir),
        transform=eval_transform,
        is_test=True
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=os.cpu_count() or 2,
        pin_memory=True
    )

    all_predictions = []
    all_extracted_ids = []

    with torch.no_grad():
        for images, ids in tqdm(inference_loader):
            images = images.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                
            all_predictions.extend(outputs.cpu().numpy())
            all_extracted_ids.extend(ids.numpy())

    df_submission = pd.DataFrame(all_predictions, columns=sample_sub.columns[1:])
    df_submission.insert(0, 'GalaxyID', all_extracted_ids)
    df_submission['GalaxyID'] = df_submission['GalaxyID'].astype(int)
    
    df_submission.to_csv(output_file, index=False)

if __name__ == '__main__':
    generate_submission(
        model_weights_path='best_model.pth',
        test_csv='data/all_ones_benchmark.csv',
        test_img_dir='data/images_test_rev1/'
    )