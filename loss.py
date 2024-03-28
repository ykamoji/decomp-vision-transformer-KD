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

        self.printed = False

    def compute_loss(self, student, inputs, return_outputs=False):

        student_inputs = inputs['pixel_values']
        student_output = self.student(student_inputs, output_hidden_states=True, output_attentions=True)

        with torch.no_grad():
            teacher_output = self.teacher(**inputs, output_hidden_states=True, output_attentions=True)

            ##TODO Losses from embedding and attentions

            if not self.printed:
                # print(teacher_output.loss)
                # print(teacher_output.logits.shape)

                print("\n")
                print("Student:")
                print("Layers:")
                print(len(student_output.hidden_states))
                print(student_output.hidden_states[0].shape)
                print("Attention layers")
                print(len(student_output.attentions))
                print(student_output.attentions[0].shape)

                print("\n")

                print("Teacher:")

                print("Layers:")
                print(len(teacher_output.hidden_states))
                print(teacher_output.hidden_states[0].shape)
                print("Attention layers")
                print(len(teacher_output.attentions))
                print(teacher_output.attentions[0].shape)

                print("\n")

                self.printed = True

        soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output.distillation_logits / self.temperature, dim=-1)

        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)

        student_target_loss = F.cross_entropy(student_output.logits, inputs['labels'])

        loss = (1. - self.alpha) * student_target_loss + self.alpha * distillation_loss

        return (loss, student_output) if return_outputs else loss
