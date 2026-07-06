import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class GalaxyZooModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.q1 = nn.Sequential(nn.Linear(in_features, 3), nn.Softmax(dim=1))
        self.q2 = nn.Sequential(nn.Linear(in_features, 2), nn.Softmax(dim=1))
        self.q3 = nn.Sequential(nn.Linear(in_features, 2), nn.Softmax(dim=1))
        self.q4 = nn.Sequential(nn.Linear(in_features, 2), nn.Softmax(dim=1))
        self.q5 = nn.Sequential(nn.Linear(in_features, 4), nn.Softmax(dim=1))
        self.q6 = nn.Sequential(nn.Linear(in_features, 2), nn.Softmax(dim=1))
        self.q7 = nn.Sequential(nn.Linear(in_features, 3), nn.Softmax(dim=1))
        self.q8 = nn.Sequential(nn.Linear(in_features, 7), nn.Softmax(dim=1))
        self.q9 = nn.Sequential(nn.Linear(in_features, 3), nn.Softmax(dim=1))
        self.q10 = nn.Sequential(nn.Linear(in_features, 3), nn.Softmax(dim=1))
        self.q11 = nn.Sequential(nn.Linear(in_features, 6), nn.Softmax(dim=1))

    def forward(self, x):
        features = self.backbone(x)
        
        out_q1 = self.q1(features) 
        
        prob_disk = out_q1[:, 1:2] 
        out_q2 = self.q2(features) * prob_disk
        
        prob_not_edge = out_q2[:, 1:2]
        out_q3 = self.q3(features) * prob_not_edge
        out_q4 = self.q4(features) * prob_not_edge
        out_q5 = self.q5(features) * prob_not_edge
        
        out_q6 = self.q6(features)
        
        prob_smooth = out_q1[:, 0:1]
        out_q7 = self.q7(features) * prob_smooth
        
        prob_odd = out_q6[:, 0:1]
        out_q8 = self.q8(features) * prob_odd
        
        out_q9 = self.q9(features)
        out_q10 = self.q10(features)
        out_q11 = self.q11(features)

        return torch.cat([
            out_q1, out_q2, out_q3, out_q4, out_q5, out_q6, 
            out_q7, out_q8, out_q9, out_q10, out_q11
        ], dim=1)