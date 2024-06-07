# file description: This file contains the code for the Concatenation model

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseConcatModel(nn.Module):
    def __init__(self, model_2d: nn.Module, model_3d: nn.Module):
        super(BaseConcatModel, self).__init__()
        self.model_2d = model_2d
        self.model_3d = model_3d

    def freeze_2d(self):
        # Freeze parameters in 2d backbone models
        for param in self.model_2d.parameters():
            param.requires_grad = False

    def freeze_3d(self):
        # Freeze parameters in 3d backbone models
        for param in self.model_3d.parameters():
            param.requires_grad = False


class ConcatModel1(BaseConcatModel):
    def __init__(self, model_2d: nn.Module, model_3d: nn.Module, input_size: int = 512, num_classes: int = 2):
        super(ConcatModel1, self).__init__(model_2d, model_3d)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, batch):
        B, _, _, _ = batch["PA_image"].shape
        self.model_2d(batch["PA_image"])
        self.model_3d(batch["CT_image"])
        f2 = self.model_2d.classifier_model.f.view(B, -1)
        f3 = self.model_3d.f.view(B, -1)
        return self.fc(torch.cat((f2, f3), dim=1))


class ConcatModel2(BaseConcatModel):
    def __init__(self, model_2d: nn.Module, model_3d: nn.Module, input_size: int = 512, num_classes: int = 2):
        super(ConcatModel2, self).__init__(model_2d, model_3d)

        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, batch):
        B = batch["PA_image"].shape[0]
        self.model_2d(batch["PA_image"])
        self.model_3d(batch["CT_image"])
        f2 = self.model_2d.hf.view(B, -1)
        f2 = self.relu(f2)
        f3 = self.model_3d.f.view(B, -1)
        f3 = self.relu(f3)
        concatenated_features = torch.cat((f2, f3), dim=1)
        x = self.fc1(concatenated_features)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ConcatModel3(BaseConcatModel):
    def __init__(self, model_2d: nn.Module, model_3d: nn.Module, input_size: int = 512, num_classes: int = 2):
        super(ConcatModel3, self).__init__(model_2d, model_3d)

        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, batch):
        B, _, _, _ = batch["PA_image"].shape
        self.model_2d(batch["PA_image"])
        self.model_3d(batch["CT_image"])
        f2 = self.model_2d.classifier_model.f.view(B, -1)
        f3 = self.model_3d.f.view(B, -1)

        # Concatenate the features
        x = torch.cat((f2, f3), dim=1)

        # Pass through the additional fully connected layer with ReLU activation
        # x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout2(x)
        # Pass through the final fully connected layer for classification
        output = self.fc2(x)
        return output


class ConcatModel4(BaseConcatModel):
    def __init__(self, model_2d: nn.Module, model_3d: nn.Module, input_size: int = 512, num_classes: int = 2):
        super(ConcatModel4, self).__init__(model_2d, model_3d)

        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, 4096)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, batch):
        B, _, _, _ = batch["PA_image"].shape
        self.model_2d(batch["PA_image"])
        self.model_3d(batch["CT_image"])
        f2 = self.model_2d.classifier_model.f.view(B, -1)
        f3 = self.model_3d.f.view(B, -1)

        # Concatenate the features
        x = torch.cat((f2, f3), dim=1)

        # Pass through the additional fully connected layer with ReLU activation
        # x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout2(x)
        # Pass through the final fully connected layer for classification
        output = self.fc2(x)
        return output


class ConcatModel4_dropout(BaseConcatModel):
    def __init__(self, model_2d: nn.Module, model_3d: nn.Module, input_size: int = 512, num_classes: int = 2):
        super(ConcatModel4_dropout, self).__init__(model_2d, model_3d)

        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, 4096)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, batch):
        B, _, _, _ = batch["PA_image"].shape
        self.model_2d(batch["PA_image"])
        self.model_3d(batch["CT_image"])
        f2 = self.model_2d.classifier_model.f.view(B, -1)
        f3 = self.model_3d.f.view(B, -1)

        # Concatenate the features
        x = torch.cat((f2, f3), dim=1)

        # Pass through the additional fully connected layer with ReLU activation
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        # Pass through the final fully connected layer for classification
        output = self.fc2(x)
        return output


class ConcatModel5(BaseConcatModel):
    def __init__(self, model_2d: nn.Module, model_3d: nn.Module, input_size: int = 512, num_classes: int = 2):
        super(ConcatModel5, self).__init__(model_2d, model_3d)

        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, 8)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(8, num_classes)

        # xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, batch):
        B = batch["PA_image"].shape[0]
        self.model_2d(batch["PA_image"])
        self.model_3d(batch["CT_image"])
        f2 = self.model_2d.classifier_model.out.view(B, -1)
        f3 = self.model_3d.out.view(B, -1)

        # Concatenate the features
        x = torch.cat((f2, f3), dim=1)

        # Pass through the additional fully connected layer with ReLU activation
        # x = self.dropout1(x)
        # x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout2(x)
        # Pass through the final fully connected layer for classification
        output = self.fc2(x)
        return output


class ConcatModel5(BaseConcatModel):
    def __init__(self, model_2d: nn.Module, model_3d: nn.Module, input_size: int = 512, num_classes: int = 2):
        super(ConcatModel5, self).__init__(model_2d, model_3d)

        self.fc1 = nn.Linear(input_size, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, num_classes)

    def forward(self, batch):
        B = batch["PA_image"].shape[0]
        self.model_2d(batch["PA_image"])
        self.model_3d(batch["CT_image"])
        f2 = self.model_2d.classifier_model.out.view(B, -1)
        f3 = self.model_3d.out.view(B, -1)

        # Concatenate the features
        x = torch.cat((f2, f3), dim=1)

        # Pass through the additional fully connected layer with ReLU activation
        # x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout2(x)
        # Pass through the final fully connected layer for classification
        output = self.fc2(x)
        return output


class ConcatModel6(BaseConcatModel):
    def __init__(self, model_2d: nn.Module, model_3d: nn.Module, input_size: int = 512, num_classes: int = 2):
        super(ConcatModel6, self).__init__(model_2d, model_3d)

        self.dropout1 = nn.Dropout(0.5)
        self.conv1x1 = nn.Conv2d(32, 1, 1)
        self.fc1 = nn.Linear(input_size, 4096)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, batch):
        B, _, _, _ = batch["PA_image"].shape
        self.model_2d(batch["PA_image"])
        self.model_3d(batch["CT_image"])
        f2 = self.model_2d.classifier_model.f
        f2 = self.relu(f2)
        f2 = self.conv1x1(f2).view(B, -1)
        f3 = self.model_3d.f.view(B, -1)

        # Concatenate the features
        x = torch.cat((f2, f3), dim=1)

        # Pass through the additional fully connected layer with ReLU activation
        # x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout2(x)
        # Pass through the final fully connected layer for classification
        output = self.fc2(x)
        return output


class ConcatModel7(BaseConcatModel):
    def __init__(self, model_2d: nn.Module, model_3d: nn.Module, input_size: int = 128, num_classes: int = 2):
        super(ConcatModel7, self).__init__(model_2d, model_3d)

        self.fc3 = nn.Linear(2048, 64)
        self.fc4 = nn.Linear(64 * 32, 64)
        self.conv1x1 = nn.Conv2d(32, 1, 1)
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, batch):
        B, _, _, _ = batch["PA_image"].shape
        self.model_2d(batch["PA_image"])
        self.model_3d(batch["CT_image"])

        f2 = self.model_2d.classifier_model.f
        f2 = self.relu(f2)
        f2 = self.conv1x1(f2).view(B, -1)
        f2 = self.fc4(f2)

        f3 = self.model_3d.f.view(B, -1)
        f3 = self.fc3(f3)

        # Concatenate the features
        x = torch.cat((f2, f3), dim=1)

        # Pass through the additional fully connected layer with ReLU activation
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)

        # Pass through the final fully connected layer for classification
        output = self.fc2(x)
        return output


class ConcatModel7_dropout(BaseConcatModel):
    def __init__(self, model_2d: nn.Module, model_3d: nn.Module, input_size: int = 512, num_classes: int = 2):
        super(ConcatModel7_dropout, self).__init__(model_2d, model_3d)

        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(64 * 32, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.conv1x1 = nn.Conv2d(32, 1, 1)
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, batch):
        B, _, _, _ = batch["PA_image"].shape
        self.model_2d(batch["PA_image"])
        self.model_3d(batch["CT_image"])

        f2 = self.model_2d.classifier_model.f
        f2 = self.relu(f2)
        f2 = self.conv1x1(f2).view(B, -1)
        f2 = self.fc4(f2)

        f3 = self.model_3d.f.view(B, -1)
        f3 = self.relu(self.fc3(f3))

        # Concatenate the features
        x = torch.cat((f2, f3), dim=1)

        # Pass through the additional fully connected layer with ReLU activation
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        # Pass through the final fully connected layer for classification
        output = self.fc2(x)
        return output


class ConcatModel8(BaseConcatModel):
    def __init__(self, model_2d: nn.Module, model_3d: nn.Module, input_size: int = 32, num_classes: int = 2):
        super(ConcatModel8, self).__init__(model_2d, model_3d)

        self.fc3 = nn.Linear(2048, 16)
        self.fc4 = nn.Linear(64 * 32, 16)
        self.conv1x1 = nn.Conv2d(32, 1, 1)
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, batch):
        B = batch["PA_image"].shape[0]
        self.model_2d(batch["PA_image"])
        self.model_3d(batch["CT_image"])

        f2 = self.model_2d.classifier_model.f
        f2 = self.relu(f2)
        f2 = self.conv1x1(f2).view(B, -1)
        f2 = self.fc4(f2)

        f3 = self.model_3d.f.view(B, -1)
        f3 = self.fc3(f3)

        # Concatenate the features
        x = torch.cat((f2, f3), dim=1)

        # Pass through the additional fully connected layer with ReLU activation
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)

        # Pass through the final fully connected layer for classification
        output = self.fc2(x)
        return output
