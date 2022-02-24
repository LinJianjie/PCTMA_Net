use_additional_Encoder = True
use_shift_data = True
inside_zero_one = False
seg_lan = 132
use_max_pool = True
use_l_adv = False
use_gt = True
select_gripper = True
select_robot = True
use_cos_similarity = False
# Data preprocessing
sos = 0
eos = 1
windows_size = 50
sliding_size = 10
max_len_windows = 200
seq_len_clr = windows_size + 2
use_fft = False
use_margin = False
margin = 1.0
size_average = True
test_anomaly_offset = 100
use_vote_accuracy = True

# Evaluate
use_max_threshold = False
# GANomaly
w_recon_gan_evaluate = 0.0# 1.0 0.00005
w_consistence_gan_evaluate = 0.0  # 1.0
w_adv_gan_evaluate = 1  # 0.05

use_max_gan = True
# Evaluate
use_predictive = False
