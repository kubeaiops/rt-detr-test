from ultralytics import RTDETR
import torch

def train_custom(): 
    model = RTDETR('model/rtdetr-l.pt')
    # Path to your custom dataset
    #data = "coco128.yaml"
    data="/Users/jerrylee/Works/rt-detr-sample/dataset/coco128/data.yaml"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cpu only will be extremly slow
    # Start training
    model.train(data=data, epochs=2, imgsz=640, device=device)

    # Save model
    save_path = "model/custom.pt"
    torch.save(model.state_dict(), save_path)

