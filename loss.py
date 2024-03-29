from transformers import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationTrainer(Trainer):
    def __init__(self,
                 teacher_model=None,
                 student_model=None,
                 temperature=None,
                 alpha=None,
                 distillation_token=False,
                 student_loss_fn=F.cross_entropy,
                 distillation_type='soft',
                 *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.distillation_loss_fun = nn.KLDivLoss(reduction="batchmean")

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

        self.distillation_token = distillation_token
        self.student_loss_fn = student_loss_fn
        self.distillation_type = distillation_type

        self.printed = False

    def _distillation_loss(self, teacher_output, student_output):

        student_tokens = student_output.logits
        if self.distillation_token:
            student_tokens = student_output.distillation_logits

        if self.distillation_type == 'soft':

            soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
            soft_student = F.log_softmax(student_tokens / self.temperature, dim=-1)

            return self.distillation_loss_fun(soft_student, soft_teacher) * (self.temperature ** 2)

        elif self.distillation_type == 'hard':

            return F.cross_entropy(student_tokens, teacher_output.logits.argmax(dim=1))

    def _student_loss(self, student_output, labels):

        if self.distillation_token:
            return self.student_loss_fn(student_output.logits, labels)
        else:
            return student_output.loss

    ## TODO: MSE loss for the layers
    ##  For distillation with distillation tokens, remove the distil token from the tensor
    ##  refer https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT/task_distill.py#L935
    def _layer_loss(self, teacher_layers, student_layers):
        return torch.zeros(1)

    ## TODO: MSE loss for the attn
    ##  For distillation with distillation tokens, remove the distil token from the tensor
    ##  refer https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT/task_distill.py#L935
    def _attn_loss(self, teacher_attn, student_attn):
        return torch.zeros(1)

    def _print_layer_shapes(self, teacher_output, student_output):

        if not self.printed:

            for model_name, logits, hidden_layers, attn_layers in zip(["Teacher", "Student"],
                                                                      [teacher_output.logits, student_output.logits],
                                                                      [teacher_output.hidden_states,
                                                                       student_output.hidden_states],
                                                                      [student_output.attentions,
                                                                       teacher_output.attentions]):
                print(f"\n{model_name}:")
                print(f"Logits Shape = {logits.shape}")
                print(f"Hidden Layers:\nDepth = {len(hidden_layers)}\nShape = {hidden_layers[0].shape}")
                print(f"Attention Layers:\nDepth = {len(attn_layers)}\nShape = {attn_layers[0].shape}")

            print("\n")

            self.printed = True

    def compute_loss(self, student, inputs, return_outputs=False):

        student_inputs = inputs
        if self.distillation_token:
            student_inputs = inputs['pixel_values']

        student_output = self.student(student_inputs, output_hidden_states=True, output_attentions=True)

        with torch.no_grad():
            teacher_output = self.teacher(**inputs, output_hidden_states=True, output_attentions=True)

        self._print_layer_shapes(teacher_output, student_output)

        distillation_loss = self._distillation_loss(teacher_output, student_output)

        student_loss = self._student_loss(student_output, inputs['labels'])

        loss = (1. - self.alpha) * student_loss + self.alpha * distillation_loss

        # loss += self._attn_loss(teacher_output.attentions, student_output.attentions)

        # loss += self._layer_loss(teacher_output.hidden_states, student_output.hidden_states)

        return (loss, student_output) if return_outputs else loss
