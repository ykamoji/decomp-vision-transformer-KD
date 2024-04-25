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
                 use_attribution_loss=False,
                 use_attention_loss=False,
                 use_hidden_loss=False,
                 *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.distillation_loss_fun = nn.KLDivLoss(reduction="sum", log_target=True)

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

        self.use_attribution_loss = use_attribution_loss
        self.use_attention_loss = use_attention_loss
        self.use_hidden_loss = use_hidden_loss
        self.attribution_loss_fn = nn.MSELoss()

        self.printed = False

    def _distillation_loss(self, teacher_output, student_output):

        student_tokens = student_output.logits
        if self.distillation_token:
            student_tokens = student_output.distillation_logits

        if self.distillation_type == 'soft':

            soft_teacher = F.log_softmax(teacher_output.logits / self.temperature, dim=-1)
            soft_student = F.log_softmax(student_tokens / self.temperature, dim=-1)

            return self.distillation_loss_fun(soft_student, soft_teacher) * (
                        self.temperature ** 2) / student_tokens.numel()

        elif self.distillation_type == 'hard':

            return F.cross_entropy(student_tokens, teacher_output.logits.argmax(dim=1))

    def _student_loss(self, student_output, labels):

        if self.distillation_token:
            return self.student_loss_fn(student_output.cls_logits, labels)
        else:
            return student_output.loss

    ## TODO: MSE loss for the layers
    ##  For distillation with distillation tokens, remove the distil token from the tensor
    ##  refer https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT/task_distill.py#L935
    def _layer_loss(self, teacher_layers, student_layers):
        return 0

    ## TODO: MSE loss for the attn
    ##  For distillation with distillation tokens, remove the distil token from the tensor
    ##  refer https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT/task_distill.py#L935
    def _attn_loss(self, teacher_atts, student_atts):
        # return torch.zeros(1)
        att_loss = 0.
        loss_mse = nn.MSELoss()

        teacher_layer_num = len(teacher_atts)
        student_layer_num = len(student_atts)
        assert teacher_layer_num % student_layer_num == 0
        layers_per_block = int(teacher_layer_num / student_layer_num)
        # layers_per_block = 1

        new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                            for i in range(student_layer_num)]

        for student_att, teacher_att in zip(student_atts, new_teacher_atts):
            student_att = torch.where(student_att <= -1e2,
                                      torch.zeros_like(student_att),
                                      student_att)
            teacher_att = torch.where(teacher_att <= -1e2,
                                      torch.zeros_like(teacher_att),
                                      teacher_att)
            #student_att shape torch.Size([8, 12, 198, 198])
            # student_att shape torch.Size([8, 12, 197, 197])
            #TODO is the last extra value is due to atrribution loss? I just removed the second one
            tmp_loss = loss_mse(student_att[:,:,:1:,:1:], teacher_att)
            att_loss += tmp_loss
        return att_loss

    def _process_attribution(self, attr):
        num_layers = len(attr)
        attribution = torch.stack([attr[i][4] for i in range(num_layers)]).squeeze()
        if attribution.ndim == 3: attribution = attribution.unsqueeze(0)
        attribution = attribution / attribution.max(dim=3)[0].unsqueeze(3)
        return attribution

    def _attribution_loss(self, teacher_attr, student_attr):
        teacher_attr = self._process_attribution(teacher_attr)
        student_attr = self._process_attribution(student_attr)

        teacher_attr = teacher_attr[:, :, 0, :]
        student_attr = student_attr[:, :, 0, :]
        if self.distillation_token:
            # student_attr_without_dist = torch.empty_like(student_attr)[:,:,:-1,:-1]
            # student_attr_without_dist[:,:,0,:] = student_attr[:,:,0,1:]
            # student_attr_without_dist[:,:,1:,:] = student_attr[:,:,2:,1:]
            # student_attr = student_attr_without_dist
            student_attr = student_attr[:,:, 1:]

        return self.attribution_loss_fn(teacher_attr, student_attr)

    def _print_layer_shapes(self, teacher_output, student_output):

        if not self.printed:

            for model_name, logits, hidden_layers, attn_layers, attr_layers in zip(["Teacher", "Student"],
                                                                      [teacher_output.logits,
                                                                       student_output.logits],
                                                                      [teacher_output.hidden_states,
                                                                       student_output.hidden_states],
                                                                      [teacher_output.attentions,
                                                                       student_output.attentions],
                                                                      [teacher_output.attributions,
                                                                       student_output.attributions]):
                print(f"\n{model_name}:")
                print(f"Logits Shape = {logits.shape}")

                if self.use_hidden_loss:
                    print(f"Hidden Layers:\nDepth = {len(hidden_layers)}\nShape = {hidden_layers[0].shape}")

                if self.use_attention_loss:
                    print(f"Attention Layers:\nDepth = {len(attn_layers)}\nShape = {attn_layers[0].shape}")

                if self.use_attribution_loss:
                    print(f"Attributions Layers:\nDepth = {len(attr_layers)}\nShape = {attr_layers[0][4].shape}")

            print("\n")

            self.printed = True

    def compute_loss(self, student, inputs, return_outputs=False):

        student_inputs = inputs

        if self.distillation_token:
            student_inputs = {'pixel_values': inputs['pixel_values']}

        kwargs = {}
        s_kwargs = {}
        if self.is_in_train:

            if self.use_hidden_loss:
                kwargs = {**kwargs, **{"output_hidden_states":True}}

            if self.use_attention_loss:
                kwargs = {**kwargs, **{"output_attentions": True}}

            if self.use_attribution_loss:
                kwargs = {**kwargs,**{"output_hidden_states":True, "output_attentions": True,
                                      "output_norms": True, "output_globenc": True}}

            # s_kwargs = {**kwargs, **{"is_student":True}}
            s_kwargs  = kwargs

        student_output = self.student(**student_inputs, **s_kwargs)

        student_loss = self._student_loss(student_output, inputs['labels'])

        if not self.is_in_train:
            return (student_loss, student_output) if return_outputs else student_loss

        with torch.no_grad():
            teacher_output = self.teacher(**inputs, **kwargs)

        self._print_layer_shapes(teacher_output, student_output)

        distillation_loss = self._distillation_loss(teacher_output, student_output)

        loss = (1. - self.alpha) * student_loss + self.alpha * distillation_loss

        if self.use_attention_loss:
            loss += self._attn_loss(teacher_output.attentions, student_output.attentions)
            print(f"l_attn {self._attn_loss(teacher_output.attentions, student_output.attentions)}")
        if self.use_hidden_loss:
            loss += self._layer_loss(teacher_output.hidden_states, student_output.hidden_states)


        if self.use_attribution_loss:
            loss += self._attribution_loss(teacher_output.attributions, student_output.attributions)

        return (loss, student_output) if return_outputs else loss
