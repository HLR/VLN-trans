# ============================================
# Configuration Sections
#   - Changing: Changing parameters
#   - Unittest: Run after adding new features,
#               before commit/push.
# ============================================


# Changing====================================

# teacher pid 68793 train-teacher.log
args = '--feedback_method teacher ' \
       '--n_iters 40000 '
'--result_dir tasks/R2R/exps/1results/ ' \
'--snapshot_dir tasks/R2R/exps/1snapshots/ ' \
'--plot_dir tasks/R2R/exps/1plots/ '

# student pid 77018 train-student.log
args = '--n_iters 60000 ' \
       '--result_dir tasks/R2R/exps/1results/ ' \
       '--snapshot_dir tasks/R2R/exps/1snapshots/ ' \
       '--plot_dir tasks/R2R/exps/1plots/ '

# FEATURE_STORE, FEATURE_SIZE = 'img_features/bottom_up', 2048
# '--result_dir tasks/R2R/exps/results/ '\
# '--snapshot_dir tasks/R2R/exps/snapshots/ '\
# '--plot_dir tasks/R2R/exps/plots/ '

# FEATURE_STORE, FEATURE_SIZE = 'img_features/ResNet-152-imagenet.tsv+' \
#                                   'img_features/bottom_up', 4096
#
# '--result_dir tasks/R2R/exps/2results/ '\
# '--snapshot_dir tasks/R2R/exps/2snapshots/ '\
# '--plot_dir tasks/R2R/exps/2plots/ '
#
# FEATURE_STORE, FEATURE_SIZE = 'img_features/ResNet-152-imagenet.tsv+' \
#                                   'img_features/bottom_up', 4096
#
# '--result_dir tasks/R2R/exps/3results/ '\
# '--snapshot_dir tasks/R2R/exps/3snapshots/ '\
# '--plot_dir tasks/R2R/exps/3plots/ '\
# '--n_iters_resume 36300 '\
# '--learning_rate 1e-5 '\
# '--sc_after 258 '\
# '--n_iters 80000 '

# train-student-len40.log
args = '--max_episode_len 40 ' \
       '--result_dir tasks/R2R/exps/4results/ ' \
       '--snapshot_dir tasks/R2R/exps/4snapshots/ ' \
       '--plot_dir tasks/R2R/exps/4plots/ ' \
       '--n_iters 50000 '

# sc+'--monotonic_sc
args = '--max_episode_len 40 ' \
       '--result_dir tasks/R2R/exps/SC2results/ ' \
       '--snapshot_dir tasks/R2R/exps/SC2snapshots/ ' \
       '--plot_dir tasks/R2R/exps/SC2plots/ ' \
       '--learning_rate 0.00001 ' \
       '--monotonic_sc True ' \
       '--n_iters_resume 45000 ' \
       '--sc_after 320 ' \
       '--n_iters 80000 '
print('copy from 4snapshots')

# sc
args = '--max_episode_len 40 ' \
       '--result_dir tasks/R2R/exps/SCresults/ ' \
       '--snapshot_dir tasks/R2R/exps/SCsnapshots/ ' \
       '--plot_dir tasks/R2R/exps/SCplots/ ' \
       '--learning_rate 0.00001 ' \
       '--monotonic_sc False ' \
       '--n_iters_resume 45000 ' \
       '--sc_after 320 ' \
       '--n_iters 80000 '
print('copy from 4snapshots')

# train-student-len60.log pid 28675
args = '--max_episode_len 60 ' \
       '--result_dir tasks/R2R/exps/6results/ ' \
       '--snapshot_dir tasks/R2R/exps/6snapshots/ ' \
       '--plot_dir tasks/R2R/exps/6plots/ ' \
       '--n_iters 70000 '

# train-student-len40-pano.log pid 42684
args = '--max_episode_len 40 ' \
       '--panoramic True ' \
       '--result_dir tasks/R2R/exps/4Presults/ ' \
       '--snapshot_dir tasks/R2R/exps/4Psnapshots/ ' \
       '--plot_dir tasks/R2R/exps/4Pplots/ ' \
       '--n_iters 65000 '

# sc-student-len60.log pid 10050
args = '--max_episode_len 60 ' \
       '--result_dir tasks/R2R/exps/SC6results/ ' \
       '--snapshot_dir tasks/R2R/exps/SC6snapshots/ ' \
       '--plot_dir tasks/R2R/exps/SC6plots/ ' \
       '--learning_rate 0.00001 ' \
       '--monotonic_sc False ' \
       '--n_iters_resume 56400 ' \
       '--sc_after 401 ' \
       '--n_iters 65000 '
print('copy from 6snapshots')

# train-student-len60-pano-bi.log pid 128727
args = '--bidirectional True ' \
       '--max_episode_len 60 ' \
       '--panoramic True ' \
       '--result_dir tasks/R2R/exps/6Pbresults/ ' \
       '--snapshot_dir tasks/R2R/exps/6Pbsnapshots/ ' \
       '--plot_dir tasks/R2R/exps/6Pbplots/ ' \
       '--n_iters 60000 '

# train-student-len40-bi.log pid xiujun
args = '--bidirectional True ' \
       '--max_episode_len 40 ' \
       '--n_iters 60000 '

# train-student-len40-pano.log pid 42684
args = '--bidirectional True ' \
       '--max_episode_len 40 ' \
       '--result_dir tasks/R2R/exps/4Presults/ ' \
       '--snapshot_dir tasks/R2R/exps/4Psnapshots/ ' \
       '--plot_dir tasks/R2R/exps/4Pplots/ ' \
       '--n_iters 60000 '

# train-student-len60-bi.log pid 48224
args = '--bidirectional True ' \
       '--max_episode_len 60 ' \
       '--result_dir tasks/R2R/exps/6bresults/ ' \
       '--snapshot_dir tasks/R2R/exps/6bsnapshots/ ' \
       '--plot_dir tasks/R2R/exps/6bplots/ ' \
       '--n_iters 60000 '

# train-student-len40-pano-bi.log pid 118377
args = '--bidirectional True ' \
       '--max_episode_len 40 ' \
       '--panoramic True ' \
       '--result_dir tasks/R2R/exps/4Pbresults/ ' \
       '--snapshot_dir tasks/R2R/exps/4Pbsnapshots/ ' \
       '--plot_dir tasks/R2R/exps/4Pbplots/ ' \
       '--n_iters 60000 '

# student-pano-act.log pid 12952
args = '--bidirectional True ' \
       '--max_episode_len 20 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresults/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshots/ ' \
       '--plot_dir tasks/R2R/exps/Nplots/ ' \
       '--n_iters 60000 '

# TODO:
#  1. clean snapshots
# student-pano-act-ctrlf.log pid 36389
args = '--max_episode_len 20 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/CFresults/ ' \
       '--snapshot_dir tasks/R2R/exps/CFsnapshots/ ' \
       '--plot_dir tasks/R2R/exps/CFplots/ ' \
       '--ctrl_feature True '

# student-pano-act-ctrlf-g.log pid 51910 ABORT
args = '--max_episode_len 20 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/CF_Gresults/ ' \
       '--snapshot_dir tasks/R2R/exps/CF_Gsnapshots/ ' \
       '--plot_dir tasks/R2R/exps/CF_Gplots/ ' \
       '--ctrl_feature True ' \
       '--use_glove True'

# student-pano-act-ctrlf-b.log pid deleted
args = '--max_episode_len 20 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/CF_Bresults/ ' \
       '--snapshot_dir tasks/R2R/exps/CF_Bsnapshots/ ' \
       '--plot_dir tasks/R2R/exps/CF_Bplots/ ' \
       '--ctrl_feature True ' \
       '--encoder_type bert ' \
       '--max_input_length 90 ' \
       '--batch_size 50 ' \
       '--learning_rate 0.00001 ' \
       '--n_iters 150000 '

# Todo: clean
#  move log to aux20.log
# student-pano-act-ctrlf-len10.log pid 10306
args = '--panoramic True ' \
    '--action_space -1 ' \
    '--result_dir tasks/R2R/exps/CFresults-len10-aux20/ ' \
    '--snapshot_dir tasks/R2R/exps/CFsnapshots-len10-aux20/ ' \
    '--plot_dir tasks/R2R/exps/CFplots-len10-aux20/ ' \
    '--ctrl_feature True ' \
    '--max_episode_len 10 ' \
    '--aux_n_iters 20 '

# Todo: clean
# student-pano-act-ctrlf-len10-aux10.log pid 20525
args = '--panoramic True ' \
    '--action_space -1 ' \
    '--result_dir tasks/R2R/exps/CFresults-len10-aux10/ ' \
    '--snapshot_dir tasks/R2R/exps/CFsnapshots-len10-aux10/ ' \
    '--plot_dir tasks/R2R/exps/CFplots-len10-aux10/ ' \
    '--ctrl_feature True ' \
    '--max_episode_len 10 ' \
    '--aux_n_iters 10 '

# student-pano-act-ctrlf-len10-aux5.log pid 30958
args = '--panoramic True ' \
    '--action_space -1 ' \
    '--result_dir tasks/R2R/exps/CFresults-len10-aux5/ ' \
    '--snapshot_dir tasks/R2R/exps/CFsnapshots-len10-aux5/ ' \
    '--plot_dir tasks/R2R/exps/CFplots-len10-aux5/ ' \
    '--ctrl_feature True ' \
    '--max_episode_len 10 ' \
    '--aux_n_iters 5 '

## START TO USE LARGER LOSS WITHOUT DIVIDING EPISODE LENGTH FROM HERE
# student-pano-act-ctrlf-len8-aux5_5.log pid 34933
args = '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/CFresults-len8-aux5_5/ ' \
       '--snapshot_dir tasks/R2R/exps/CFsnapshots-len8-aux5_5/ ' \
       '--plot_dir tasks/R2R/exps/CFplots-len8-aux5_5/ ' \
       '--ctrl_feature True ' \
       '--max_episode_len 8 ' \
       '--aux_n_iters 5 ' \
       '--aux_ratio 5 '

# student-pano-act-ctrlf-len8-aux10_10.log pid 119382
args = '--panoramic True ' \
    '--action_space -1 ' \
    '--result_dir tasks/R2R/exps/CFresults-len8-aux10_10/ ' \
    '--snapshot_dir tasks/R2R/exps/CFsnapshots-len8-aux10_10/ ' \
    '--plot_dir tasks/R2R/exps/CFplots-len8-aux10_10/ ' \
    '--ctrl_feature True ' \
    '--max_episode_len 8 ' \
    '--aux_n_iters 10 ' \
    '--aux_ratio 10 ' \
    '--aux_ratio 5 '

# student-pano-act-ctrlf-len8-aux2_1.log pid 18470 111608 67501
args = '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/CFresults-len8-aux2_1/ ' \
       '--snapshot_dir tasks/R2R/exps/CFsnapshots-len8-aux2_1/ ' \
       '--plot_dir tasks/R2R/exps/CFplots-len8-aux2_1/ ' \
       '--ctrl_feature True ' \
       '--max_episode_len 8 ' \
       '--aux_n_iters 2 ' \
       '--aux_ratio 1 ' \
       '--n_iters_resume 58500 '\
       '--n_iters 120000 '

# student-pano-act-len8.log pid 130239
args = '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/CFresults-len8/ ' \
       '--snapshot_dir tasks/R2R/exps/CFsnapshots-len8/ ' \
       '--plot_dir tasks/R2R/exps/CFplots-len8/ ' \
       '--max_episode_len 8 '

# sc-pano-act-len8  pid 15090 27202 75036
args = '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/SC_N_len8results/ ' \
       '--snapshot_dir tasks/R2R/exps/SC_N_len8snapshots/ ' \
       '--plot_dir tasks/R2R/exps/SC_N_len8plots/ ' \
       '--max_episode_len 8 ' \
       '--sc_learning_rate 0.000001 ' \
       '--n_iters_resume 49900 ' \
       '--sc_after 50000 ' \
       '--n_iters 120000 '

# todo: clean deconv setting1
# student-pano-act-ctrlf-len8-aux2_1-deconv.log pid 16723
# args = '--panoramic True ' \
#     '--action_space -1 ' \
#     '--result_dir tasks/R2R/exps/CFresults-len8-aux2_1-deconv/ ' \
#     '--snapshot_dir tasks/R2R/exps/CFsnapshots-len8-aux2_1-deconv/ ' \
#     '--plot_dir tasks/R2R/exps/CFplots-len8-aux2_1-deconv/ ' \
#     '--ctrl_feature True ' \
#     '--max_episode_len 8 ' \
#     '--ctrl_f_net deconv ' \
#     '--n_iters 120000' \
#     '--aux_n_iters 2 ' \
#     '--aux_ratio 1 '

# student-pano-act-ctrlf-len8-aux2_1-deconv.log pid 26731
args = '--panoramic True ' \
    '--action_space -1 ' \
    '--result_dir tasks/R2R/exps/CFresults-len8-aux2_1-deconv/ ' \
    '--snapshot_dir tasks/R2R/exps/CFsnapshots-len8-aux2_1-deconv/ ' \
    '--plot_dir tasks/R2R/exps/CFplots-len8-aux2_1-deconv/ ' \
    '--ctrl_feature True ' \
    '--max_episode_len 8 ' \
    '--ctrl_f_net deconv ' \
    '--n_iters 120000' \
    '--aux_n_iters 2 ' \
    '--aux_ratio 1 '

# student-pano-act-ctrlf-len8-aux5_1-deconv.log pid 38770
args = '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/CFresults-len8-aux5_1-deconv-set2/ ' \
       '--snapshot_dir tasks/R2R/exps/CFsnapshots-len8-aux5_1-deconv-set2/ ' \
       '--plot_dir tasks/R2R/exps/CFplots-len8-aux5_1-deconv-set2/ ' \
       '--ctrl_feature True ' \
       '--max_episode_len 8 ' \
       '--ctrl_f_net deconv ' \
       '--n_iters 120000 ' \
       '--aux_n_iters 5 ' \
       '--aux_ratio 1 '

# student-pano-act-ctrlf-len8-aux5_1.log pid 67564
args = '--panoramic True ' \
    '--action_space -1 ' \
    '--result_dir tasks/R2R/exps/CFresults-len8-aux5_1/ ' \
    '--snapshot_dir tasks/R2R/exps/CFsnapshots-len8-aux5_1/ ' \
    '--plot_dir tasks/R2R/exps/CFplots-len8-aux5_1/ ' \
    '--ctrl_feature True ' \
    '--max_episode_len 8 ' \
    '--n_iters 120000 ' \
    '--aux_n_iters 5 ' \
    '--aux_ratio 1 '

# student-pano-act-len8-b.log pid shoob/3142 (cleaned)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/BNresults/ ' \
       '--snapshot_dir tasks/R2R/exps/BNsnapshots/ ' \
       '--plot_dir tasks/R2R/exps/BNplots/ ' \
       '--encoder_type bert ' \
       '--max_input_length 90 ' \
       '--batch_size 20 ' \
       '--learning_rate 0.00002 ' \
       '--n_iters 200000 '

# student-pano-act-len8-stop.log pid  76092
args = '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresults-len8-stop/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshots-len8-stop/ ' \
       '--plot_dir tasks/R2R/exps/Nplots-len8-stop/ ' \
       '--max_episode_len 8 ' \
       '--learn_stop True '

# student-pano-act-len8-subaug.log pid
args = '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresults-len8-subaug/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshots-len8-subaug/ ' \
       '--plot_dir tasks/R2R/exps/Nplots-len8-subaug/ ' \
       '--max_episode_len 8 ' \
       '--subgoal_aug True '


# student-pano-act-len8-gpt.log pid rainier/
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/GNresults/ ' \
       '--snapshot_dir tasks/R2R/exps/GNsnapshots/ ' \
       '--plot_dir tasks/R2R/exps/GNplots/ ' \
       '--encoder_type gpt ' \
       '--max_input_length 80 ' \
       '--batch_size 20 ' \
       '--learning_rate 0.00002 ' \
       '--n_iters 200000 '

# student-pano-act-len8-gpt-lstm.log pid shoob/9887 (cleaned)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/GNresults-lstm/ ' \
       '--snapshot_dir tasks/R2R/exps/GNsnapshots-lstm/ ' \
       '--plot_dir tasks/R2R/exps/GNplots-lstm/ ' \
       '--encoder_type gpt ' \
       '--max_input_length 80 ' \
       '--batch_size 20 ' \
       '--learning_rate 0.00002 ' \
       '--n_iters 200000 ' \
       '--top_lstm True '

# student-pano-act-len8-mean.log pid rainier/104386
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresult-l8-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshot-l8-mean/ ' \
       '--plot_dir tasks/R2R/exps/Nplot-l8-mean/ ' \
       '--att_ctx_merge mean '

# student-pano-act-len8-max.log pid rainier/100402
args = '--max_episode_len 8 ' \
    '--panoramic True ' \
    '--action_space -1 ' \
    '--result_dir tasks/R2R/exps/Nresult-l8-max/ ' \
    '--snapshot_dir tasks/R2R/exps/Nsnapshot-l8-max/ ' \
    '--plot_dir tasks/R2R/exps/Nplot-l8-max/ ' \
    '--att_ctx_merge max '

# stu-pano-act-len8-cat.log pid rainier/103828
args = '--max_episode_len 8 ' \
    '--panoramic True ' \
    '--action_space -1 ' \
    '--result_dir tasks/R2R/exps/Nresult-l8-cat/ ' \
    '--snapshot_dir tasks/R2R/exps/Nsnapshot-l8-cat/ ' \
    '--plot_dir tasks/R2R/exps/Nplot-l8-cat/ ' \
    '--att_ctx_merge cat '

# student-pano-act-len8-mean-aux.log pid rainier/117528
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresult-l8-mean-aux/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshot-l8-mean-aux/ ' \
       '--plot_dir tasks/R2R/exps/Nplot-l8-mean-aux/ ' \
       '--att_ctx_merge mean '\
       '--ctrl_feature True ' \
       '--aux_n_iters 2 ' \
       '--aux_ratio 1 ' \
       '--n_iters 100000 '

# student-pano-act-len8-bi-mean.log pid rainier/37434
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresult-len8-bi-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshot-len8-bi-mean/ ' \
       '--plot_dir tasks/R2R/exps/Nplot-len8-bi-mean/ ' \
       '--att_ctx_merge mean '\
       '--bidirectional True '

# student-pano-act-len8-sum.log pid rainier/85082
args = '--max_episode_len 8 ' \
    '--panoramic True ' \
    '--action_space -1 ' \
    '--result_dir tasks/R2R/exps/Nresult-l8-sum/ ' \
    '--snapshot_dir tasks/R2R/exps/Nsnapshot-l8-sum/ ' \
    '--plot_dir tasks/R2R/exps/Nplot-l8-sum/ ' \
    '--att_ctx_merge sum '

# student-pano-act-len8-noshare.log pid rainier/3426
args = '--max_episode_len 8 ' \
    '--panoramic True ' \
    '--action_space -1 ' \
    '--result_dir tasks/R2R/exps/Nresult-l8-noshare/ ' \
    '--snapshot_dir tasks/R2R/exps/Nsnapshot-l8-noshare/ ' \
    '--plot_dir tasks/R2R/exps/Nplot-l8-noshare/ ' \
    '--att_ctx_merge mean ' \
    '--multi_share False '

# student-pano-act-len8-pre-sc.log pid rainier/55621
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresult-l8-pre-sc/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshot-l8-pre-sc/ ' \
       '--plot_dir tasks/R2R/exps/Nplot-l8-pre-sc/ ' \
       '--att_ctx_merge mean ' \
       '--multi_share True ' \
       '--use_pretrain True ' \
       '--pretrain_n_sentences 1 ' \
       '--pretrain_n_iters 50000 ' \
       '--sc_after 100000 ' \
       '--n_iters 150000'

# for debugging
# student-pano-act-len8-pre-sc.log pid rainier/
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/test/ ' \
       '--snapshot_dir tasks/R2R/exps/test/ ' \
       '--plot_dir tasks/R2R/exps/test/ ' \
       '--att_ctx_merge mean ' \
       '--multi_share True ' \
       '--use_pretrain True ' \
       '--pretrain_splits literal_speaker_train,train ' \
       '--pretrain_n_sentences 4 ' \
       '--pretrain_n_iters 200 ' \
       '--sc_after 400 ' \
       '--n_iters 1000'

# for debugging
# student-pano-act-len8-pre-sc.log pid rainier/
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/test/ ' \
       '--snapshot_dir tasks/R2R/exps/test/ ' \
       '--plot_dir tasks/R2R/exps/test/ ' \
       '--att_ctx_merge mean ' \
       '--multi_share True ' \
       '--use_pretrain True ' \
       '--pretrain_n_sentences 1 ' \
       '--pretrain_n_iters 200 ' \
       '--sc_after 400 ' \
       '--n_iters 1000'

# Nstu-l8-N4-after800.log pid rainier/28526
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresult-l8-N4-after800/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshots-l8-N4-after800/ ' \
       '--plot_dir tasks/R2R/exps/Nplot-l8-pre6-N4-after800/ ' \
       '--att_ctx_merge mean ' \
       '--multi_share True ' \
       '--use_pretrain True ' \
       '--pretrain_n_sentences 4 ' \
       '--pretrain_splits literal_speaker_train,train ' \
       '--pretrain_n_iters 80000 ' \
       '--n_iters 120000'

# Nstu-l8-N3-after500-1000-1500.log pid rainier/91721
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresult-l8-N3-after500-1000-1500/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshots-l8-N3-after-500-1000-1500/ ' \
       '--plot_dir tasks/R2R/exps/Nplot-l8-pre6-N3-after-500-1000-1500/ ' \
       '--att_ctx_merge mean ' \
       '--multi_share True ' \
       '--use_pretrain True ' \
       '--pretrain_n_sentences 3 ' \
       '--pretrain_splits literal_speaker_data_augmentation_paths,' \
                         'sample_seed10_literal_speaker_data_augmentation_paths,' \
                         'sample_seed20_literal_speaker_data_augmentation_paths,' \
                         'train ' \
       '--pretrain_n_iters 50000 ' \
       '--sc_after 100000 ' \
       '--n_iters 150000'

# pycharm
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8/ ' \
       '--encoder_type transformer ' \
       '--n_iters 200000 ' \
       '--transformer_emb_size 512 ' \
       '--transformer_d_ff 1024 ' \
       '--decoder_init False ' \
       '--transformer_num_layers 1'

# TNstu-pano-act-len8-256-512-2-mean.log pid rainier/67452  (clean)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 200000 ' \
       '--transformer_emb_size 256 ' \
       '--transformer_d_ff 512 ' \
       '--decoder_init False ' \
       '--transformer_num_layers 2'


# TNstu-pano-act-len8-512-1204-2-mean.log pid rainier/103921  (clean)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-512-1204-2-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-512-1204-2-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-512-1204-2-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 200000 ' \
       '--transformer_emb_size 512 ' \
       '--transformer_d_ff 1024 ' \
       '--transformer_num_layers 2 ' \
       '--decoder_init False ' \
       '--score_name spl_unseen '

# TNstu-pano-act-len8-512-1024-1-mean.log pid rainier/11031  (clean)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-512-1204-1-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-512-1204-1-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-512-1204-1-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 200000 ' \
       '--transformer_emb_size 512 ' \
       '--transformer_d_ff 1024 ' \
       '--transformer_num_layers 1 ' \
       '--decoder_init False ' \
       '--score_name spl_unseen '

# TNstu-pano-act-len8-512-1024-3-mean.log pid shoob/16019 (cleaned)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-512-1204-3-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-512-1204-3-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-512-1204-3-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 200000 ' \
       '--transformer_emb_size 512 ' \
       '--transformer_d_ff 1024 ' \
       '--transformer_num_layers 3 ' \
       '--score_name spl_unseen ' \
       '--decoder_init False ' \
       '--save_room True '

# TNstu-pano-act-len8-512-1024-3-h8-mean.log pid rainier/125562 (clean)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-512-1024-3-h8-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-512-1024-3-h8-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-512-1024-3-h8-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 200000 ' \
       '--transformer_emb_size 512 ' \
       '--transformer_d_ff 1024 ' \
       '--transformer_num_layers 3 ' \
       '--heads 8 ' \
       '--decoder_init False ' \
       '--score_name spl_unseen '

# TNstu-pano-act-len8-768-1024-2-h12-mean.log pid rainier/ (clean)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-768-1024-2-h12-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-768-1024-2-h12-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-768-1024-2-h12-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 200000 ' \
       '--transformer_emb_size 768 ' \
       '--transformer_d_ff 1024 ' \
       '--transformer_num_layers 2 ' \
       '--heads 12 ' \
       '--score_name spl_unseen ' \
       '--decoder_init False ' \
       '--save_room True'

# TNstu-pano-act-len8-520-1024-3-h10-mean.log pid shoob/26798  (cleaned)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-520-1024-3-h10-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-520-1024-3-h10-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-520-1024-3-h10-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 200000 ' \
       '--transformer_emb_size 520 ' \
       '--transformer_d_ff 1024 ' \
       '--transformer_num_layers 3 ' \
       '--heads 10 ' \
       '--decoder_init False ' \
       '--score_name spl_unseen '

# TNstu-pano-act-len8-510-1024-4-h10-mean.log pid shoob/5241 cleaned
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-510-1024-4-h10-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-510-1024-4-h10-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-510-1024-4-h10-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 70000 ' \
       '--transformer_emb_size 510 ' \
       '--transformer_d_ff 1024 ' \
       '--transformer_num_layers 4 ' \
       '--heads 10 ' \
       '--score_name spl_unseen ' \
       '--decoder_init False ' \
       '--save_room True '
#
# TNstu-pano-act-len8-516-1024-3-h12-mean.log pid shoob/27142 (cleaned)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-516-1024-3-h12-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-516-1024-3-h12-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-516-1024-3-h12-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 200000 ' \
       '--transformer_emb_size 516 ' \
       '--transformer_d_ff 1024 ' \
       '--transformer_num_layers 3 ' \
       '--heads 12 ' \
       '--decoder_init False ' \
       '--score_name spl_unseen '


# TNstu-pano-act-len8-512-1024-4-mean.log pid shoob/15937 (cleaned)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-512-1024-4-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-512-1024-4-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-512-1024-4-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 200000 ' \
       '--transformer_emb_size 512 ' \
       '--transformer_d_ff 1024 ' \
       '--transformer_num_layers 4 ' \
       '--decoder_init False ' \
       '--score_name spl_unseen '


# TNstu-pano-act-len8-512-2048-4-mean.log pid shoob/16539 (cleaned)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-512-2048-4-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-512-2048-4-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-512-2048-4-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 300000 ' \
       '--transformer_emb_size 512 ' \
       '--transformer_d_ff 2048 ' \
       '--transformer_num_layers 4 ' \
       '--decoder_init False ' \
       '--score_name spl_unseen '


# TNstu-pano-act-len8-512-2048-8-mean.log pid shoob/6232 (cleaned)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-512-2048-8-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-512-2048-8-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-512-2048-8-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 500000 ' \
       '--transformer_emb_size 512 ' \
       '--transformer_d_ff 2048 ' \
       '--transformer_num_layers 8 ' \
       '--decoder_init False ' \
       '--score_name spl_unseen '

# TNstu-pano-act-len8-768-1024-2-h12-mean.log pid rainier/78080 (clean)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-768-1024-2-h12-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-768-1024-2-h12-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-768-1024-2-h12-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 200000 ' \
       '--transformer_emb_size 768 ' \
       '--transformer_d_ff 1024 ' \
       '--transformer_num_layers 2 ' \
       '--heads 12 ' \
       '--score_name spl_unseen ' \
       '--decoder_init False ' \
       '--save_room True'

# TNstu-pano-act-len8-768-1024-3-h12-mean.log pid bonanza/21165 (clean)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-768-1024-3-h12-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-768-1024-3-h12-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-768-1024-3-h12-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 200000 ' \
       '--transformer_emb_size 768 ' \
       '--transformer_d_ff 1024 ' \
       '--transformer_num_layers 3 ' \
       '--heads 12 ' \
       '--score_name spl_unseen ' \
       '--decoder_init False ' \
       '--save_room True'

# TNstu-pano-act-len8-516-516-4-h12-mean.log pid shoob/5001  (cleaned)
args = '--max_episode_len 8 ' \
    '--panoramic True ' \
    '--action_space -1 ' \
    '--result_dir tasks/R2R/exps/TNresults-l8-516-516-4-h12-mean/ ' \
    '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-516-516-4-h12-mean/ ' \
    '--plot_dir tasks/R2R/exps/TNplots-l8-516-516-4-h12-mean/ ' \
    '--encoder_type transformer ' \
    '--att_ctx_merge mean ' \
    '--n_iters 100000 ' \
    '--transformer_emb_size 516 ' \
    '--transformer_d_ff 516 ' \
    '--transformer_num_layers 4 ' \
    '--heads 12 ' \
    '--decoder_init False ' \
    '--score_name spl_unseen '

# TNstu-pano-act-len8-516-1204-4-h12-mean.log pid shoob/2353  (cleaned)
args = '--max_episode_len 8 ' \
    '--panoramic True ' \
    '--action_space -1 ' \
    '--result_dir tasks/R2R/exps/TNresults-l8-516-1024-4-h12-mean/ ' \
    '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-516-1024-4-h12-mean/ ' \
    '--plot_dir tasks/R2R/exps/TNplots-l8-516-1024-4-h12-mean/ ' \
    '--encoder_type transformer ' \
    '--att_ctx_merge mean ' \
    '--n_iters 100000 ' \
    '--transformer_emb_size 516 ' \
    '--transformer_d_ff 1024 ' \
    '--transformer_num_layers 4 ' \
    '--heads 12 ' \
    '--decoder_init False ' \
    '--score_name spl_unseen '

# TNstu-pano-act-len8-300-1024-4-h12-mean.log pid shoob/2567 cleaned
args = '--max_episode_len 8 ' \
    '--panoramic True ' \
    '--action_space -1 ' \
    '--result_dir tasks/R2R/exps/TNresults-l8-300-1024-4-h12-mean/ ' \
    '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-300-1024-4-h12-mean/ ' \
    '--plot_dir tasks/R2R/exps/TNplots-l8-300-1024-4-h12-mean/ ' \
    '--encoder_type transformer ' \
    '--att_ctx_merge mean ' \
    '--n_iters 70000 ' \
    '--transformer_emb_size 300 ' \
    '--transformer_d_ff 1024 ' \
    '--transformer_num_layers 4 ' \
    '--heads 12 ' \
    '--score_name spl_unseen ' \
       '--decoder_init False ' \
    '--save_room True '

# TNstu-pano-act-len8-512-1024-4-h16-mean.log pid rainier/
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-512-1024-4-h16-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-512-1024-4-h16-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-512-1024-4-h16-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 70000 ' \
       '--transformer_emb_size 512 ' \
       '--transformer_d_ff 1024 ' \
       '--transformer_num_layers 4 ' \
       '--heads 16 ' \
       '--score_name spl_unseen ' \
       '--decoder_init False ' \
       '--save_room True '

# TNstu-pano-act-len8-516-1024-5-h12-mean.log pid shoob/13766 (cleaned)
args = '--max_episode_len 8 ' \
    '--panoramic True ' \
    '--action_space -1 ' \
    '--result_dir tasks/R2R/exps/TNresults-l8-516-1024-5-h12-mean/ ' \
    '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-516-1024-5-h12-mean/ ' \
    '--plot_dir tasks/R2R/exps/TNplots-l8-516-1024-5-h12-mean/ ' \
    '--encoder_type transformer ' \
    '--att_ctx_merge mean ' \
    '--n_iters 100000 ' \
    '--transformer_emb_size 516 ' \
    '--transformer_d_ff 1024 ' \
    '--transformer_num_layers 5 ' \
    '--heads 12 ' \
    '--decoder_init False ' \
    '--score_name spl_unseen '

# TNstu-pano-act-len8-516-1204-6-h12-mean.log pid shoob/22966  (cleaned)
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-516-1024-6-h12-mean/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-516-1024-6-h12-mean/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-516-1024-6-h12-mean/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 100000 ' \
       '--transformer_emb_size 516 ' \
       '--transformer_d_ff 1024 ' \
       '--transformer_num_layers 6 ' \
       '--heads 12 ' \
       '--decoder_init False ' \
       '--score_name spl_unseen '

# TNstu-pano-act-len8-516-1204-4-h12-mean-rainier.log rainier/80082 cleaned
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/TNresults-l8-516-1024-4-h12-mean-rainier/ ' \
       '--snapshot_dir tasks/R2R/exps/TNsnapshots-l8-516-1024-4-h12-mean-rainier/ ' \
       '--plot_dir tasks/R2R/exps/TNplots-l8-516-1024-4-h12-mean-rainier/ ' \
       '--encoder_type transformer ' \
       '--att_ctx_merge mean ' \
       '--n_iters 100000 ' \
       '--transformer_emb_size 516 ' \
       '--transformer_d_ff 1024 ' \
       '--transformer_num_layers 4 ' \
       '--heads 12 ' \
       '--score_name spl_unseen ' \
       '--save_room True '

# region after error-corrected
# Nstu-l8-mean-v2.log pid  bonanza/6588
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresult-l8-mean-v2/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshot-l8-mean-v2/ ' \
       '--plot_dir tasks/R2R/exps/Nplot-l8-mean-v2/ ' \
       '--att_ctx_merge mean '
# Nstu-l8-sum-v2.log pid  bonanza/7671
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresult-l8-sum-v2/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshot-l8-sum-v2/ ' \
       '--plot_dir tasks/R2R/exps/Nplot-l8-sum-v2/ ' \
       '--att_ctx_merge sum '

# Nstu-l8-sum-nodi.log pid  rainier/87251 running
args = '--max_episode_len 8 ' \
    '--panoramic True ' \
    '--action_space -1 ' \
    '--result_dir tasks/R2R/exps/Nresult-l8-sum-nodi/ ' \
    '--snapshot_dir tasks/R2R/exps/Nsnapshot-l8-sum-nodi/ ' \
    '--plot_dir tasks/R2R/exps/Nplot-l8-sum-nodi/ ' \
    '--att_ctx_merge sum ' \
    '--decoder_init False '
# Nstu-l8-max-v2.log pid  bonanza/9634
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresult-l8-max-v2/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshot-l8-max-v2/ ' \
       '--plot_dir tasks/R2R/exps/Nplot-l8-max-v2/ ' \
       '--att_ctx_merge max '
# Nstu-l8-cat-v2.log pid  bonanza/5689
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresult-l8-cat-v2/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshot-l8-cat-v2/ ' \
       '--plot_dir tasks/R2R/exps/Nplot-l8-cat-v2/ ' \
       '--att_ctx_merge cat '
# endregion

args = '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/test/ ' \
       '--snapshot_dir tasks/R2R/exps/test/ ' \
       '--plot_dir tasks/R2R/exps/test/ ' \
       '--ctrl_feature True ' \
       '--max_episode_len 8 ' \
       '--aux_n_iters 5 ' \
       '--aux_ratio 5 ' \
       '--n_iters_resume 55700 ' \
       '--train False ' \
       '--val_splits val_seen,val_unseen,test '

# for transformer
arg_choices = {'--encoder_type':'transformer',
               '--heads':[2,4,8],
               '--bidirectional': ['True'],
               '--att_ctx_merge mean ':['mean','None'],
               '--transformer_emb_size':[256,512,768],
               '--transformer_d_ff': [512, 1024, 2048],
               '--transformer_num_layers':[1,2,3,4],
               '--score_name':['spl_unseen','spl_sum','sr_sum']}

# FROM THIS VERSION ENCODER AND DECODER USE DIFFERENT HIDDEN SIZE ARGUMENT!!
# SO PREVIOUS BIDIRECTIONAL MODEL MAY ENCOUNTER LOADING ERROR!!!! IF SO, CHECKOUT PREVIOUS COMMIT!
# Nstu-len8-mean-bi1024.log pid 25803
args = '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresult-len8-mean-bi-1024/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshot-len8-mean-bi-1024/ ' \
       '--plot_dir tasks/R2R/exps/Nplot-len8-mean-bi-1024/ ' \
       '--max_episode_len 8 ' \
       '--bidirectional True ' \
       '--att_ctx_merge mean ' \
       '--enc_hidden_size 512 ' \
       '--hidden_size 1024 '

# Nstu-len8-mean-bi1024-512.log pid 25803  gpu1
args = '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresult-len8-mean-bi-1024-512/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshot-len8-mean-bi-1024-512/ ' \
       '--plot_dir tasks/R2R/exps/Nplot-len8-mean-bi-1024-512/ ' \
       '--max_episode_len 8 ' \
       '--bidirectional True ' \
       '--att_ctx_merge mean ' \
       '--enc_hidden_size 512 ' \
       '--hidden_size 512 '

args = '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/test/ ' \
       '--snapshot_dir tasks/R2R/exps/best/ ' \
       '--plot_dir tasks/R2R/exps/test/ ' \
       '--max_episode_len 8 ' \
       '--bidirectional False ' \
       '--att_ctx_merge mean ' \
       '--enc_hidden_size 512 ' \
       '--hidden_size 512 ' \
       '--n_iters_resume 81800 ' \
       '--train False '

"CUDA_VISIBLE_DEVICES=0 python ./tasks/R2R/train.py --max_episode_len 8 --panoramic True --result_dir tasks/R2R/exps/SEncoder/newDU/basresults/ --snapshot_dir tasks/R2R/exps/SEncoder/newDU/bassnapshots/ --plot_dir tasks/R2R/exps/SEncoder/newDU/basplots/ --action_space -1 --n_iters 50000 --att_ctx_merge mean"  # pid 64874 gpu0

#
# for plot attention of 5671
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/test/ ' \
       '--snapshot_dir tasks/R2R/exps/best/ ' \
       '--plot_dir tasks/R2R/exps/test/ ' \
       '--att_ctx_merge mean ' \
       '--n_iters_resume 81800 ' \
       '--train False ' \
       '--val_splits debug1 ' \
       '--batch_size 1 '

# MultiBertEncoder for multiple sentences
args = '--feedback_method teacher ' \
       '--result_dir /home/xql/Source/Subgoal/tasks/R2R/exps/test_bert_mean/results/ ' \
       '--snapshot_dir /home/xql/Source/Subgoal/tasks/R2R/exps/test_bert_mean/snapshots/ ' \
       '--plot_dir /home/xql/Source/Subgoal/tasks/R2R/exps/test_bert_mean/plots/ ' \
       '--ss_n_iters 20000 ' \
       '--dropout_ratio 0.4 ' \
       '--dec_h_type vc --schedule_ratio 0.3 ' \
       '--optm Adam --clip_gradient_norm 0 --log_every 64 ' \
       '--action_space -1 ' \
       '--train_score_name sr_unseen ' \
       '--n_iters 40000 ' \
       '--enc_hidden_size 1024 --hidden_size 1024 ' \
       '--bidirectional True ' \
       '--batch_size 10 ' \
       '--encoder_type bert --top_lstm True --transformer_update False ' \
       '--att_ctx_merge mean '
# Unittests===================================

# Test bidirectional
args = '--bidirectional True ' \
       '--max_episode_len 60 ' \
       '--result_dir tasks/R2R/exps/test_results/ ' \
       '--snapshot_dir tasks/R2R/exps/6bsnapshots/ ' \
       '--plot_dir tasks/R2R/exps/test_plots/ ' \
       '--n_iters_resume 41600 ' \
       '--train False '
# should be 0.498 & 0.2576


# Test panoramic
args = '--max_episode_len 40 ' \
       '--panoramic True ' \
       '--result_dir tasks/R2R/exps/test_results/ ' \
       '--snapshot_dir tasks/R2R/exps/4Psnapshots/ ' \
       '--plot_dir tasks/R2R/exps/test_plots/ ' \
       '--n_iters_resume 33400 ' \
       '--train False '
# should be 0.5098 & 0.304


# Test sc+monotonic_sc
args = '--max_episode_len 40 ' \
       '--result_dir tasks/R2R/exps/test_results/ ' \
       '--snapshot_dir tasks/R2R/exps/SC2snapshots/ ' \
       '--plot_dir tasks/R2R/exps/test_plots/ ' \
       '--sc_learning_rate 0.00001 ' \
       '--sc_after 38900 ' \
       '--sc_reward_scale 5 ' \
       '--sc_discouted_immediate_r_scale 0.01 ' \
       '--sc_length_scale 0.01 '  \
       '--monotonic_sc True ' \
       '--n_iters_resume 45000 ' \
       '--sc_after 45100 ' \
       '--n_iters 65000 ' \
       '--train True '  # CAUTION
print('copy from 4snapshots')
# should be 0.4588 & 0.2546


# Test pano action_space
args = '--bidirectional True ' \
       '--max_episode_len 20 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/test_results/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshots/ ' \
       '--plot_dir tasks/R2R/exps/test_plots/ ' \
       '--n_iters_resume 56300 ' \
       '--train False '
# should be 0.432 & 0.247


# student-pano-act-len8-mean.log pid rainier/104386
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/test_results/ ' \
       '--snapshot_dir tasks/R2R/exps/test/ ' \
       '--plot_dir tasks/R2R/exps/test_plots/ ' \
       '--att_ctx_merge mean '


# Nstu-l8-N4-N3-after500-1000-1500.log pid rainier/
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/test/ ' \
       '--snapshot_dir tasks/R2R/exps/test/ ' \
       '--plot_dir tasks/R2R/exps/test/ ' \
       '--att_ctx_merge mean ' \
       '--multi_share True ' \
       '--use_pretrain False ' \
       '--pretrain_n_sentences 3 ' \
       '--pretrain_splits literal_speaker_data_augmentation_paths,' \
                     'sample_seed10_literal_speaker_data_augmentation_paths,' \
                     'sample_seed20_literal_speaker_data_augmentation_paths,' \
                     'train ' \
       '--pretrain_n_iters 100 ' \
       '--sc_after 38900 ' \
       '--sc_reward_scale 5 ' \
       '--sc_discouted_immediate_r_scale 0.01 ' \
       '--sc_length_scale 0.01 '  \
       '--n_iters 39000 ' \
       '--sc_learning_rate 0.00005 ' \
       '--train True ' \
       '--n_iters_resume 38800 ' \
       '--score_name spl_unseen '

# student-pano-act-len8-mean.log pid rainier/104386
args = '--max_episode_len 8 ' \
       '--panoramic True ' \
       '--action_space -1 ' \
       '--result_dir tasks/R2R/exps/Nresult-l8-mean-v2(copy)/ ' \
       '--snapshot_dir tasks/R2R/exps/Nsnapshot-l8-mean-v2(copy)/ ' \
       '--plot_dir tasks/R2R/exps/Nplot-l8-mean-v2(copy)/ ' \
       '--att_ctx_merge mean ' \
       '--n_iters_resume 41900 ' \
       '--sc_after 0 ' \
       '--pretrain_score_name sr_sum ' \
       '--train_score_name sr_unseen ' \
       '--sc_score_name spl_unseen '
