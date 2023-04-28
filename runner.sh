### train ###
CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py --train \
--config human36m_diffpose_uvxyz_cpn.yml --batch_size 1024 \
--model_pose_path checkpoints/gcn_xyz_cpn.pth \
--doc human36m_diffpose_uvxyz_cpn --exp exp --ni \
>exp/human36m_diffpose_uvxyz_cpn.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 python main_diffpose_frame.py --train \
--config human36m_diffpose_uvxyz_gt.yml --batch_size 1024 \
--model_pose_path checkpoints/gcn_xyz_gt.pth \
--doc human36m_diffpose_uvxyz_gt --exp exp --ni \
>exp/human36m_diffpose_uvxyz_gt.out 2>&1 &

### test ###
CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py \
--config human36m_diffpose_uvxyz_cpn.yml --batch_size 1024 \
--model_pose_path checkpoints/gcn_xyz_cpn.pth \
--model_diff_path checkpoints/diffpose_uvxyz_cpn.pth \
--doc t_human36m_diffpose_uvxyz_cpn --exp exp --ni \
>exp/t_human36m_diffpose_uvxyz_cpn.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 python main_diffpose_frame.py \
--config human36m_diffpose_uvxyz_gt.yml --batch_size 1024 \
--model_pose_path checkpoints/gcn_xyz_gt.pth \
--model_diff_path checkpoints/diffpose_uvxyz_gt.pth \
--doc t_human36m_diffpose_uvxyz_gt --exp exp --ni \
>exp/t_human36m_diffpose_uvxyz_gt.out 2>&1 &