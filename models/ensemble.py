import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, models_list, num_classes):
        super().__init__()
        self.models = nn.ModuleList(models_list)
        self.num_models = len(models_list)
        self.num_classes = num_classes

        # Final classification layer after concatenation
        self.fc = nn.Linear(num_classes * self.num_models, num_classes)

    def forward(self, mel_spec, mfcc, lofar):
        outputs = []
        for model in self.models:
            out, _ = model(mel_spec, mfcc, lofar)
            outputs.append(torch.softmax(out, dim=1))
        # Concatenate outputs across models
        combined = torch.cat(outputs, dim=1)
        # Final classification as weighted sum
        final_output = self.fc(combined)
        return final_output

# Usage: Instantiate ensemble with list of trained models and number of classes
