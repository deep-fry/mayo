import numpy as np
import math
from itertools import product

from mayo.override.base import OverriderBase
from mayo.override.quantize import Recentralizer, FloatingPointQuantizer
from mayo.override.quantize import ShiftQuantizer
from mayo.session.retrain.base import RetrainBase
from mayo.log import log


class GlobalRetrain(RetrainBase):
    def forward_policy(self, floor_epoch):
        self.best_ckpt = 'retrain-' + str(self.retrain_cnt) + '-' \
            + str(floor_epoch)
        self.save_checkpoint(self.best_ckpt)
        self._cp_epoch = floor_epoch
        self.retrain_cnt += 1
        self.log_thresholds(self.loss_avg, self.acc_avg)
        self.profile_associated_vars()
        self.variables_refresh()
        self.reset_num_epochs()
        threshold = self.targets.members[0].thresholds[0]
        log.info('update threshold to {}, working on {}'.format(
            threshold, self.target_layer))
        return True

    def backward_policy(self):
        # if did not reach min scale
        tmp_tv = self.targets.members[0]
        if tmp_tv.scale > 0:
            scale_check = self._fetch_scale() > tmp_tv.min_scale
            threshold_check = tmp_tv.end_thresholds[0] > tmp_tv.thresholds[0]
        elif tmp_tv.scale == 0:
            scale_check = False
            threshold_check = False
        else:
            scale_check = self._fetch_scale() < tmp_tv.min_scale
            threshold_check = tmp_tv.end_thresholds[0] < tmp_tv.thresholds[0]

        if scale_check and threshold_check:
            # retrace the best ckpt
            self.load_checkpoint(self.best_ckpt)
            self._decrease_scale()
            log.info(
                'Decreasing scale to {}, threshold is {}...'
                .format(self._fetch_scale(), tmp_tv.thresholds[0]))
            self.reset_num_epochs()
            return True
        # stop if reach min scale
        else:
            self.targets.cont = []
            thresholds = tmp_tv.thresholds[0]
            scale = tmp_tv.scale
            log.info(
                'All layers done, final threshold is {}'
                .format(thresholds - scale))
            if not threshold_check:
                log.info('threshold meets its minimum')
            if not scale_check:
                log.info('scale meets its minimum')
            log.info(
                'Overrider is done, model stored at {}.'
                .format(self.best_ckpt))
            self.reset_num_epochs()
            return False

    def variables_refresh(self):
        update_flag = False
        for item in self.targets.members:
            self.variable_refresh(item)
            if isinstance(item.av, OverriderBase):
                item.av.should_update = True
                update_flag = True
        tmp = self.targets.show_associates()[0]
        # check if it is and only is floating point
        check_shift = self._check_cls(tmp, ShiftQuantizer)
        check_float = self._check_cls(tmp, FloatingPointQuantizer) \
            and not check_shift
        # in global retrain, thresholds are globally the same
        w = self.targets.members[0].thresholds[0]
        if check_float:
            self.allocate_exp_mantissa(w)
        if check_shift:
            for item in self.targets.members:
                av_before = self.run(item.av.before)
                if not item.mean_quantizer:
                    raise ValueError('Mean quantizer not defined!')
                self._update_mean_quantizer(av_before, item.mean_quantizer, w)
        if update_flag:
            self.overriders_update()

    def _check_cls(self, av, cls):
        if isinstance(av, cls):
            return True
        if isinstance(av, Recentralizer):
            if isinstance(av.quantizer, cls):
                return True
        return False

    def allocate_exp_mantissa(self, width, overflow_rate=0.0):
        '''
        Description:
            a hack and a specialized method for floating point quantizer.
        Problem:
            floating point quantization has two searching parameters: exponent
            width and mantissa width, each of them has a different impact on
            accuracy.
        Solution:
            I proposed a before retraining loss check. Brute-froce search all
            the possible combinations between exponents and mantissa, pick
            the combination that has the least loss compared to full-precision
            weights
        '''
        log.info("search to allocate exp and mantissa parts")
        biases = losses = []
        for mantissa_width in range(0, int(width) + 1):
            loss = 0
            biases = []
            exponent_width = width - mantissa_width
            for item in self.targets.members:
                org_matrix = self.run(item.av.quantizer.before)
                tmp, bias = item.av.quantizer.compute_quantization_loss(
                    org_matrix[org_matrix != 0], exponent_width,
                    mantissa_width, overflow_rate)
                # accumulate loss
                loss += tmp
                # collect bias
                biases.append(bias)
            losses.append((loss, exponent_width, mantissa_width, biases))
        # find smallest loss
        _, exp_width, mantissa_width, biases = min(
            losses, key=lambda meta: meta[0])
        index = 0
        for item in self.targets.members:
            values = self.run(item.av.before)
            self._update_mean_quantizer(
                values, item.mean_quantizer, width, overflow_rate)
            self.assign(item.quantizer.mantissa_width, mantissa_width)
            self.assign(item.quantizer.exponent_bias, biases[index])
            self.assign(item.bias_quantizer.mantissa_width, mantissa_width)
            self.assign(item.bias_quantizer.exponent_bias, biases[index])
            index += 1

    def _compute_mean_exp(self, pos_mean, neg_mean, width, overflow_rate):
        max_exponent = int(2 ** width)
        for exp in range(min(-max_exponent, -10), max(max_exponent, 4)):
            max_value = 2 ** (exp + 1)
            if neg_mean > -max_value and pos_mean < max_value:
                break
        return exp

    def _update_mean_quantizer(
            self, values, variable, width, overflow_rate=0.01):
        positives = np.mean(values[values > 0])
        negatives = np.mean(values[values < 0])
        exp = self._compute_mean_exp(
            positives, negatives, width, overflow_rate)
        # shift quantizer has mantissa zero
        if not isinstance(variable, ShiftQuantizer):
            if exp > 0:
                self.assign(
                    variable.mantissa_width,
                    width - math.ceil(math.log(exp, 2)))
            else:
                self.assign(variable.mantissa_width, width)
        if exp < 0:
            self.assign(variable.exponent_bias, -exp)
        else:
            self.assign(variable.exponent_bias, 0)

    def _decrease_scale(self):
        # decrease scale factor, for quantizer, this factor might be 1
        for item in self.targets.members:
            # roll back on thresholds
            threshold = item.thresholds[0]
            scale = item.scale
            if threshold == item.start_threshold:
                raise ValueError(
                    'Thresholds failed on starting point, consider '
                    'changing starting point.')
            else:
                item.thresholds = [
                    th - scale for th in item.thresholds]
            # decrease scale
            if isinstance(scale, int) and isinstance(threshold, int):
                item.scale = int(item.scale * item.update_factor)
            else:
                item.scale = item.scale * item.update_factor
        tmp = self.targets.members[0].av
        check_floating_point = self._check_cls(tmp, FloatingPointQuantizer) \
            and not self._check_cls(tmp, ShiftQuantizer)
        for item in self.targets.members:
            # use new scale
            self.variable_refresh(item)
        if check_floating_point:
            w = self.targets.members[0].thresholds[0]
            self.allocate_exp_mantissa(w)
        self.overriders_update()

    def retrain_simple(self):
        session = self.task.session
        overriders = self.task.nets[0].overriders
        targets = self.config.retrain.parameters.target
        ranges = self.config.retrain.parameters.range
        link_width = self.config.retrain.parameters.pop('link_width', None)
        if len(ranges) == 1:
            ranges = len(targets) * ranges
        ranges = [self.parse_range(r) for r in ranges]
        q_losses = []
        items = []
        for item in product(*ranges):
            if link_width and item[link_width[0]] > item[link_width[1]]:
                continue
            q_loss = 0
            for o in overriders:
                self.assign_targets(o, targets, item)
                before, after = session.run([o.before, o.after])
                q_loss += self.quantization_loss(before, after)
            q_losses.append(q_loss)
            items.append(item)
        self.present(overriders, items, q_losses)
        return False

    def present(self, overriders, items, losses):
        sel_loss = np.min(np.array(losses))
        sel_arg = np.argmin(np.array(losses))
        formatter = ('suggested bitwidths: {}, quantize loss: {}')
        log.info(formatter.format(items[sel_arg], sel_loss))
        return
