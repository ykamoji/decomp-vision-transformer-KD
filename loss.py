from transformers import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, student_model=None, temperature=None, alpha=None, *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = 'mps'

        device = torch.device(device)
        self.teacher.to(device)
        self.teacher.eval()
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, student, inputs, return_outputs=False):
        student_output = self.student(**inputs)

        with torch.no_grad():
            teacher_output = self.teacher(**inputs)

        soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)

        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)

        student_target_loss = student_output.loss

        loss = (1. - self.alpha) * student_target_loss + self.alpha * distillation_loss
        return (loss, student_output) if return_outputs else loss
