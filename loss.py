from transformers import Trainer
from utils.featureUtils import get_device
from process_datasets import processInputs
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationTrainer(Trainer):
    def __init__(self,
                 teacher_model=None,
                 student_model=None,
                 temperature=None,
                 alpha=None,
                 student_loss_fn=F.cross_entropy,
                 configArgs=None,
                 writer=None,
                 *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.distillation_loss_fun = nn.KLDivLoss(reduction="sum", log_target=True)

        device = get_device()
        self.teacher.to(device)
        self.teacher.eval()
        self.temperature = temperature
        self.alpha = alpha
        self.configArgs = configArgs

        self.distillation_token = configArgs.Distillation.UseDistTokens
        self.student_loss_fn = student_loss_fn
        self.distillation_type = configArgs.Distillation.DistillationType

        self.use_attribution_loss = configArgs.Distillation.UseAttributionLoss
        self.use_attention_loss = configArgs.Distillation.UseAttentionLoss
        self.use_hidden_loss = configArgs.Distillation.UseHiddenLoss
        self.use_ats_loss = configArgs.Distillation.UseATSLoss
        self.attribution_loss_fn = nn.MSELoss()
        self.ats_loss_fn = nn.MSELoss()

        self.printed = False
        self.global_steps = 0
        self.writer = writer
        self.current_epoch = 0

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

    def _layer_loss(self, teacher_layers, student_layers):
        layer_loss = 0.

        embedding_loss = self._hidden_loss(student_layers[0], teacher_layers[0])

        self.tb_log('Loss/Embedding', embedding_loss.item())

        teacher_layers = teacher_layers[1:]
        student_layers = student_layers[1:]

        teacher_layer_num = len(teacher_layers)
        student_layer_num = len(student_layers)
        assert teacher_layer_num % student_layer_num == 0
        layers_per_block = int(teacher_layer_num / student_layer_num)

        new_teacher_layers = [
            teacher_layers[i * layers_per_block + layers_per_block - 1]
            for i in range(student_layer_num)]

        for student_layer, teacher_layer in zip(student_layers, new_teacher_layers):
            student_layer = torch.where(student_layer <= -1e2,
                                        torch.zeros_like(student_layer),
                                        student_layer)
            teacher_layer = torch.where(teacher_layer <= -1e2,
                                        torch.zeros_like(teacher_layer),
                                        teacher_layer)

            if self.distillation_token:
                indices = torch.tensor([0] + list(range(2, student_layer.shape[1])), device=student_layer.device)
                student_layer = torch.index_select(student_layer, 1, indices)

            layer_loss += self._hidden_loss(student_layer, teacher_layer)

        layer_loss += embedding_loss

        return layer_loss

    def _hidden_loss(self, student_layer, teacher_layer):
        loss_mse = nn.MSELoss()
        if student_layer.shape[-1] == teacher_layer.shape[-1]:
            tmp_loss = loss_mse(student_layer, teacher_layer)
        else:
            tmp_loss = loss_mse(self._layer_similarity(student_layer), self._layer_similarity(teacher_layer))
        return tmp_loss

    def _layer_similarity(self, hidden_states):
        hidden_states = hidden_states.flatten(1)
        return F.cosine_similarity(hidden_states[:,:,None], hidden_states.t()[None,:,:])

    def _attn_loss(self, teacher_atts, student_atts):
        att_loss = 0.
        loss_mse = nn.MSELoss()
        teacher_layer_num = len(teacher_atts)
        student_layer_num = len(student_atts)
        assert teacher_layer_num % student_layer_num == 0
        layers_per_block = int(teacher_layer_num / student_layer_num)

        new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                            for i in range(student_layer_num)]

        for student_att, teacher_att in zip(student_atts, new_teacher_atts):
            student_att = torch.where(student_att <= -1e2,
                                      torch.zeros_like(student_att),
                                      student_att)
            teacher_att = torch.where(teacher_att <= -1e2,
                                      torch.zeros_like(teacher_att),
                                      teacher_att)

            if self.distillation_token:
                indices = torch.tensor([0] + list(range(2, student_att.shape[3])), device=student_att.device)
                student_att = torch.index_select(student_att, 2, indices)
                student_att = torch.index_select(student_att, 3, indices)

            if student_att.shape[1] != teacher_att.shape[1]:
                tmp_loss = loss_mse(student_att.mean(1), teacher_att.mean(1))
            else:
                tmp_loss = loss_mse(student_att, teacher_att)
            att_loss += tmp_loss
        return att_loss

    def _process_attribution(self, attr):
        num_layers = len(attr)
        if type(attr[0]) is tuple:
            attribution = torch.stack([attr[i][4] for i in range(num_layers)]).squeeze()
        else:
            attribution = torch.stack([attr[i] for i in range(num_layers)]).squeeze()
        # if attribution.ndim == 3: attribution = attribution.unsqueeze(0)
        # attribution = attribution / attribution.max(dim=3)[0].unsqueeze(3)
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
            student_attr = student_attr[:, :, 1:]

        if teacher_attr.shape[0] != student_attr.shape[0]:
            attr_loss = 0.
            teacher_layer_num = len(teacher_attr)
            student_layer_num = len(student_attr)
            assert teacher_layer_num % student_layer_num == 0
            layers_per_block = int(teacher_layer_num / student_layer_num)
            new_teacher_atts = [teacher_attr[i * layers_per_block + layers_per_block - 1]
                                for i in range(student_layer_num)]
            for student_att, teacher_att in zip(student_attr, new_teacher_atts):
                student_att = torch.where(student_att <= -1e2,
                                          torch.zeros_like(student_att),
                                          student_att)
                teacher_att = torch.where(teacher_att <= -1e2,
                                          torch.zeros_like(teacher_att),
                                          teacher_att)

                attr_loss += self.attribution_loss_fn(teacher_att, student_att)

            return attr_loss
        else:
            return self.attribution_loss_fn(teacher_attr, student_attr)

    def _ats_loss(self, teacher_ats, student_ats):
        num_layers = len(teacher_ats)
        teacher_ats = torch.stack([teacher_ats[i] for i in range(num_layers)]).squeeze()

        num_layers = len(student_ats)
        student_ats = torch.stack([student_ats[i] for i in range(num_layers)]).squeeze()

        if self.distillation_token:
            student_ats = student_ats[:, :, 1:]

        return self.ats_loss_fn(teacher_ats, student_ats)

    def _print_layer_shapes(self, teacher_output, student_output):

        if not self.printed:

            for model_name, logits, hidden_layers, attn_layers, attr_layers, ats_layers \
                    in zip(["Teacher", "Student"], [teacher_output.logits, student_output.logits],
                           [teacher_output.hidden_states, student_output.hidden_states],
                           [teacher_output.attentions, student_output.attentions],
                           [teacher_output.attributions, student_output.attributions],
                           [teacher_output.ats_attentions, student_output.ats_attentions]):

                print(f"\n{model_name}:")
                print(f"Logits Shape = {logits.shape}")

                if self.use_hidden_loss:
                    print(f"Hidden Layers:\nDepth = {len(hidden_layers)}\nShape = {hidden_layers[0].shape}")

                if self.use_attention_loss:
                    print(f"Attention Layers:\nDepth = {len(attn_layers)}\nShape = {attn_layers[0].shape}")

                if self.use_attribution_loss:
                    print(f"Attributions Layers:\nDepth = {len(attr_layers)}\nShape = {attr_layers[0].shape}")

                if self.use_ats_loss:
                    print(f"ATS Layers:\nDepth = {len(ats_layers)}\nShape = {ats_layers[0].shape}")

            print("\n")

            self.printed = True

    def tb_log(self, key, value):
        if self.current_epoch != self.state.epoch and self.state.global_step % self.state.logging_steps == 0:
            self.writer.add_scalar(key, value, global_step=self.state.global_step, walltime=5)

    def compute_loss(self, student, inputs, return_outputs=False):

        if self.configArgs.Common.DataSet.Name == 'imageNet':
            inputs = processInputs(inputs, self.configArgs.Distillation.Model)

        for k, v in inputs.items():
            inputs[k] = v.to(get_device())

        student_inputs = inputs

        if self.distillation_token:
            student_inputs = {'pixel_values': inputs['pixel_values']}

        kwargs = {}
        s_kwargs = {}
        if not return_outputs:

            if self.use_hidden_loss:
                kwargs = {**kwargs, **{"output_hidden_states": True}}

            if self.use_attention_loss:
                kwargs = {**kwargs, **{"output_attentions": True}}

            if self.use_attribution_loss:
                kwargs = {**kwargs, **{"output_hidden_states": True, "output_attentions": True,
                                       "output_norms": False, "output_globenc": True}}

            if self.use_ats_loss:
                kwargs = {**kwargs, **{"output_ats": 1}}

            s_kwargs = {**kwargs, **{"is_student":True}}
            # s_kwargs = kwargs

        student_output = self.student(**student_inputs, **s_kwargs)

        student_loss = self._student_loss(student_output, inputs['labels'])

        if return_outputs:
            return student_loss, student_output

        with torch.no_grad():
            teacher_output = self.teacher(**inputs, **kwargs)

        self._print_layer_shapes(teacher_output, student_output)

        distillation_loss = self._distillation_loss(teacher_output, student_output)

        self.tb_log('Loss/Student', student_loss)

        self.tb_log('Loss/Distillation', distillation_loss)

        loss = (1. - self.alpha) * student_loss + self.alpha * distillation_loss

        if self.use_attention_loss:
            attn_loss = self._attn_loss(teacher_output.attentions, student_output.attentions)
            self.tb_log('Loss/Attention', attn_loss.item())
            loss += attn_loss

        if self.use_hidden_loss:
            hidden_loss = self._layer_loss(teacher_output.hidden_states, student_output.hidden_states)
            self.tb_log('Loss/Hidden', hidden_loss.item())
            loss += hidden_loss

        if self.use_attribution_loss:
            attr_loss = self._attribution_loss(teacher_output.attributions, student_output.attributions)
            self.tb_log('Loss/Attribution', attr_loss.item())
            loss += attr_loss

        if self.use_ats_loss:
            ats_loss = self._ats_loss(teacher_output.ats_attentions, student_output.ats_attentions)
            self.tb_log('Loss/ATS', ats_loss.item())
            loss += ats_loss

        if self.current_epoch != self.state.epoch:
            self.current_epoch = self.state.epoch

        return loss
