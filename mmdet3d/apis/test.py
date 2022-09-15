import mmcv
import torch
from mmcv.image import tensor2imgs
from os import path as osp

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    show_pretrain=False):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    occ_data=[]
    import numpy as np
    x_coors = np.arange(0, 200)
    y_coors = np.arange(0, 200)
    X_coors, Y_coors = np.meshgrid(x_coors, y_coors)
    x = X_coors * 0.5 + 0.5 / 2 - 50
    y = Y_coors * 0.5 + 0.5 / 2 - 50
    r = np.sqrt(x ** 2 + y ** 2)
    store = {
        "occupied_bev": np.zeros((len(data_loader), 200, 200), dtype=np.int8),
        "gt_num_points_bev": np.zeros((len(data_loader), 200, 200)),
        "diff_num_points_bev": np.zeros((len(data_loader), 200, 200))}

    num_test_samples = len(data_loader)
    num_viz = 20
    period = num_test_samples//num_viz
    for i, data in enumerate(data_loader):
        if i % period:
            continue
        
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, pretrain=show_pretrain, **data)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(data, result, out_dir)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

        if show_pretrain:
            import matplotlib.pyplot as plt
            from matplotlib import ticker, cm
            import matplotlib.colors as colors
            import matplotlib.cbook as cbook

            SMALL_SIZE = 200
            MEDIUM_SIZE = 300
            BIGGER_SIZE = 400

            plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
            import numpy as np
            extent = result["point_cloud_range"][::3] + result["point_cloud_range"][1::3]

            vx, vy, vz = result["voxel_shape"]

            x_range = np.arange(extent[0] + vx / 2, extent[1] - vx / 2, vx - 1e-16)
            y_range = np.arange(extent[2] + vy / 2, extent[3] - vy / 2, vy - 1e-16)
            X, Y = np.meshgrid(x_range, y_range)

            if result["occupied_bev"] is not None:
                store["occupied_bev"][i] = result["occupied_bev"][0].detach().cpu().numpy().astype(np.int8).T
                """batch_size = result["occupied_bev"].shape[0]
                # vmin, vmax = -1, 5
                # cticks = [-1, 0, 1, 2, 3, 4, 5]
                for b in range(batch_size):
                    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(200, 100))
                    occ_bev =result["occupied_bev"][b].detach().cpu().numpy().T

                    # Even bounds give a contour-like effect:
                    bounds = np.linspace(-1.5, 5.5, 8)
                    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=7)
                    cMap = colors.ListedColormap(
                        ["w", 'limegreen', 'darkgreen', "orangered", "darkred", "gold",  "darkgoldenrod"])
                    pcm = ax1.pcolormesh(X, Y, occ_bev, norm=norm, cmap=cMap)
                    pcm2 = ax2.pcolormesh(X[50:150, 100:], Y[50:150, 100:], occ_bev[50:150, 100:], norm=norm, cmap=cMap)
                    cb = fig.colorbar(pcm, orientation='vertical')
                    cb.ax.set_xticks(cticks, ['Empty', 'True Unmasked ', 'False Unmasked', 'True Masked ', 'False Masked', 'False Fake', 'True Fake'])

                    #im = plt.imshow(occ_bev, extent=extent, vmin=vmin, vmax=vmax)
                    plt.suptitle(f"Occupied prediction, Datapoint {i}, batch {b}")
                    plt.savefig(f"occ_pred_{i}_{b}.png")
                    plt.close()
                    data_dict = {
                        "n_points": (occ_bev > -1).sum(),
                        "TN": ((occ_bev == 0) | (occ_bev == 2)).sum(),
                        "FP": ((occ_bev == 1) | (occ_bev == 3)).sum(),
                        "FN": (occ_bev == 4).sum(),
                        "TP": (occ_bev == 5).sum(),
                        "sample": i*batch_size+b,
                    }
                    data_dict["FPR"] = data_dict["FP"]/(data_dict["TN"]+data_dict["FP"])
                    data_dict["FNR"] = data_dict["FN"]/(data_dict["TP"]+data_dict["FN"])
                    data_dict["Recall"] = data_dict["TP"]/(data_dict["TP"]+data_dict["FN"])  # TPR
                    data_dict["Precision"] = data_dict["TP"]/(data_dict["TP"]+data_dict["FP"])
                    data_dict["Accuracy"] = (data_dict["TP"] + data_dict["TN"]) / data_dict["n_points"]
                    occ_data.append(data_dict)"""

            if result["gt_num_points_bev"] is not None:
                store["gt_num_points_bev"][i] = result["gt_num_points_bev"][0].detach().cpu().numpy().T
                """batch_size = result["gt_num_points_bev"].shape[0]
                for b in range(batch_size):
                    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(200, 200))
                    gt_num_points_bev = result["gt_num_points_bev"][b].detach().cpu().numpy().T


                    diff_num_points_bev = result["diff_num_points_bev"][b].detach().cpu().numpy().T
                    pred_num_points = gt_num_points_bev - diff_num_points_bev

                    vmin = gt_num_points_bev[gt_num_points_bev != 0].min()
                    assert X.shape == gt_num_points_bev.shape
                    pcm = ax1.pcolor(X, Y, gt_num_points_bev,
                                       norm=colors.LogNorm(vmin=vmin, vmax=gt_num_points_bev.max()),
                                       cmap='PuBu_r', shading='auto')
                    ax1.set_title("Ground truth")

                    pcm2 = ax2.pcolor(X, Y, pred_num_points,
                                     norm=colors.LogNorm(vmin=vmin, vmax=gt_num_points_bev.max()),
                                     cmap='PuBu_r', shading='auto')
                    ax2.set_title("Predicted")
                    pcm3 = ax3.pcolor(X[80:120, 100:140], Y[80:120, 100:140], gt_num_points_bev[80:120, 100:140],
                                     norm=colors.LogNorm(vmin=vmin, vmax=gt_num_points_bev.max()),
                                     cmap='PuBu_r', shading='auto')
                    pcm4 = ax4.pcolor(X[80:120, 100:140], Y[80:120, 100:140], pred_num_points[80:120, 100:140],
                                      norm=colors.LogNorm(vmin=vmin, vmax=gt_num_points_bev.max()),
                                      cmap='PuBu_r', shading='auto')
                    fig.colorbar(pcm, extend='max')
                    plt.title(f"Number of points per voxel BEV, Datapoint {i}, batch {b}")
                    plt.savefig(f"gt_num_points_bev{i}_{b}.png")
                    plt.close()"""
            if result["diff_num_points_bev"] is not None:
                store["diff_num_points_bev"][i] = result["diff_num_points_bev"][0].detach().cpu().numpy().T
                """batch_size = result["diff_num_points_bev"].shape[0]
                for b in range(batch_size):
                    fig = plt.figure(figsize=(100, 100))
                    diff_num_points_bev = result["diff_num_points_bev"][b].detach().cpu().numpy().T
                    diff_num_points_bev = np.abs(diff_num_points_bev)
                    vmin = diff_num_points_bev[diff_num_points_bev != 0].min()
                    vmax = diff_num_points_bev.max()
                    assert X.shape == diff_num_points_bev.shape
                    pcm = plt.pcolor(X, Y, diff_num_points_bev,
                                     norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                                     cmap='PuBu_r', shading='auto')
                    fig.colorbar(pcm, extend='both')
                    plt.title(f"Diff in predicted number of points per voxel BEV, Datapoint {i}, batch {b}")
                    plt.savefig(f"diff_num_points_bev{i}_{b}.png")
                    plt.close()"""
            if result["points"] is not None:
                # xticks = np.arange(result["point_cloud_range"][0], result["point_cloud_range"][3] + 0.000001, step=result["voxel_shape"][0])
                xticks_large = np.arange(-50, 50 + 0.000001, 0.5*16)
                xticks_small = np.arange(0, 15 + 0.000001, 0.5)
                xmask = xticks_small % 5 == 0
                xlabels = [round(xticks_small[i], 2) if xmask[i] else None for i in range(xticks_small.size)]
                # xmask = xticks % 10 == 0
                # xmask = np.diff((xticks / 10).astype(int), append=0.0) > 0
                # xlabels = [round(xticks[i], 2) if xmask[i] else None for i in range(xticks.size)]

                # yticks = np.arange(result["point_cloud_range"][1], result["point_cloud_range"][3] + 0.000001, step=result["voxel_shape"][1])
                yticks_large = np.arange(-50, 50 + 0.000001, 0.5*16)
                yticks_small = np.arange(-7.5, 7.5 + 0.000001, 0.5)
                ymask = yticks_small % 5 == 0
                ylabels = [round(yticks_small[i], 2) if ymask[i] else None for i in range(yticks_small.size)]
                # ymask = np.diff((yticks / 10).astype(int), append=0.0) > 0
                # ymask = xticks % 10 == 0
                # ylabels = [round(yticks[i], 2) if ymask[i] else None for i in range(yticks.size)]

                batch = result["points_batch"]
                gt_batch = result["gt_points_batch"]
                batch_size = int(result["gt_points_batch"].max().item()) + 1
                for b in range(batch_size):
                    points = result["points"][torch.where(batch == b)].detach().cpu().numpy()
                    gt_points = result["gt_points"][torch.where(gt_batch == b)].detach().cpu().numpy()

                    cmin = min(points[:, 2].min(), gt_points[:, 2].min())
                    cmax = min(points[:, 2].max(), gt_points[:, 2].max())
                    color = (points[:, 2] - cmin)/(cmax - cmin)
                    gt_color = (gt_points[:, 2] - cmin)/(cmax - cmin)
                    gt_mask = (gt_points[:, 0] > 0) & (gt_points[:, 0] < 15) & (gt_points[:, 1] > -7.5) & (gt_points[:, 1] < 7.5)
                    p_mask = (points[:, 0] > 0) & (points[:, 0] < 15) & (points[:, 1] > -7.5) & (points[:, 1] < 7.5)

                    masked_voxel_coors = result["masked_voxel_coors"]
                    masked_voxel_coors_set = {(m[3], m[2]) for m in masked_voxel_coors} #(x,y) pairs
                    unmasked_voxel_coors = result["unmasked_voxel_coors"]
                    unmasked_voxel_coors_set = {(int(m[3]), int(m[2])) for m in unmasked_voxel_coors} #(x,y) pairs
                    voxel_offset = np.array(result["voxel_offset"])
                    voxel_shape = np.array(result["voxel_shape"])
                    points_voxel_idx = (points[:,:2] - voxel_offset[:2] + voxel_shape[:2]/2)//voxel_shape[:2]
                    gt_points_voxel_idx = (gt_points[:,:2] - voxel_offset[:2] + voxel_shape[:2]/2)//voxel_shape[:2]
                    unmasked_gt_point_mask = np.array([True if (int(idx[0]), int(idx[1])) in unmasked_voxel_coors_set else False for idx in gt_points_voxel_idx])


                    path="/mimer/NOBACKUP/groups/snic2021-7-127/eliassv/jobs/figs_all_points/099"
                    


                    f, ax1 = plt.subplots(1, 1, figsize=(100, 100))
                    ax1.scatter(gt_points[:, 0], gt_points[:, 1], s=60, c=gt_color, label="GT")
                    ax1.set_title("Ground truth")
                    ax1.set_xticks(xticks_large)
                    ax1.set_yticks(yticks_large)
                    ax1.set_xlim([-50, 50])
                    ax1.set_ylim([-50, 50])
                    ax1.grid()
                    fig_path = osp.join(path, f"gt_points_bev{i}_{b}.pdf")
                    plt.savefig(fig_path)
                    plt.close()

                    f, ax2 = plt.subplots(1, 1, figsize=(100, 100))
                    ax2.scatter(gt_points[unmasked_gt_point_mask][:, 0], gt_points[unmasked_gt_point_mask][:, 1], s=60, c=gt_color[unmasked_gt_point_mask], label="GT")
                    ax2.set_title("Input")
                    ax2.set_xticks(xticks_large)
                    ax2.set_yticks(yticks_large)
                    ax2.set_xlim([-50, 50])
                    ax2.set_ylim([-50, 50])
                    ax2.grid()
                    fig_path = osp.join(path, f"input_points_bev{i}_{b}.pdf")
                    plt.savefig(fig_path)
                    plt.close()

                    f, ax3 = plt.subplots(1, 1, figsize=(100, 100))
                    ax3.scatter(points[:, 0], points[:, 1], s=60, c=color, label="Predicted")
                    ax3.set_title("Predicted")
                    ax3.set_xticks(xticks_large)
                    ax3.set_yticks(yticks_large)
                    ax3.set_xlim([-50, 50])
                    ax3.set_ylim([-50, 50])
                    ax3.grid()
                    fig_path = osp.join(path, f"predicted_points_bev{i}_{b}.pdf")
                    plt.savefig(fig_path)
                    plt.close()

                    f, ax4 = plt.subplots(1, 1, figsize=(100, 100))
                    ax4.scatter(gt_points[gt_mask][:, 0], gt_points[gt_mask][:, 1], s=45*45, c=gt_color[gt_mask], label="GT")
                    ax4.set_title("Ground truth")
                    ax4.set_xticks(xticks_small, xlabels)
                    ax4.set_yticks(yticks_small, ylabels)
                    ax4.set_xlim([0, 15])
                    ax4.set_ylim([-7.5, 7.5])
                    ax4.grid()
                    fig_path = osp.join(path, f"gt_points_zoom_bev{i}_{b}.pdf")
                    plt.savefig(fig_path)
                    plt.close()
                    
                    f, ax5 = plt.subplots(1, 1, figsize=(100, 100))
                    gt_mask2 = unmasked_gt_point_mask & gt_mask
                    ax5.scatter(gt_points[gt_mask2][:, 0], gt_points[gt_mask2][:, 1], s=45*45, c=gt_color[gt_mask2], label="GT")
                    ax5.set_title("Input")
                    ax5.set_xticks(xticks_small, xlabels)
                    ax5.set_yticks(yticks_small, ylabels)
                    ax5.set_xlim([0, 15])
                    ax5.set_ylim([-7.5, 7.5])
                    ax5.grid()
                    fig_path = osp.join(path, f"input_points_zoom_bev{i}_{b}.pdf")
                    plt.savefig(fig_path)
                    plt.close()

                    f, ax6 = plt.subplots(1, 1, figsize=(100, 100))
                    ax6.scatter(points[p_mask][:, 0], points[p_mask][:, 1], s=45*45, c=color[p_mask], label="Predicted")
                    ax6.set_title("Predicted")
                    ax6.set_xticks(xticks_small, xlabels)
                    ax6.set_yticks(yticks_small, ylabels)
                    ax6.set_xlim([0, 15])
                    ax6.set_ylim([-7.5, 7.5])
                    ax6.grid()
                    #f.suptitle(f"Predicted point locations, Datapoint {i}, batch {b}")
                    #path="/mimer/NOBACKUP/groups/snic2021-7-127/eliassv/jobs/figs"
                    fig_path = osp.join(path, f"predicted_points_zoom_bev{i}_{b}.pdf")
                    plt.savefig(fig_path)
                    plt.close()

        results.extend(result)

        batch_size = 1 #len(result)
        for _ in range(batch_size):
            prog_bar.update()
        """if show_pretrain and i % 10 == 0:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            SMALL_SIZE = 8
            MEDIUM_SIZE = 10
            BIGGER_SIZE = 12
            plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
            plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
            df = pd.DataFrame(occ_data)
            df[["TN", "TP", "FN", "FP"]] = df[["TN", "TP", "FN", "FP"]]/df["n_points"].to_numpy().reshape(-1, 1)
            df_merge = pd.melt(df[[
                "TN", "TP", "FN", "FP", "FPR", "FNR", "Recall", "Precision", "Accuracy"
            ]])
            sns.boxplot(x="variable", y="value", data=df_merge)
            plt.grid(axis='x')
            plt.savefig(f"occupied_metrics_{i}.png")
            plt.close()"""
        if show_pretrain and i % 100 == 0:
            for key, val in store.items():
                np.save(key+"_i", val)
    if show_pretrain:
        for key, val in store.items():
            np.save(key + "_i", val)
        """import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12
        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        df = pd.DataFrame(occ_data)
        df[["TN", "TP", "FN", "FP"]] = df[["TN", "TP", "FN", "FP"]] / df["n_points"].to_numpy().reshape(-1, 1)
        df_merge = pd.melt(df[[
            "TN", "TP", "FN", "FP", "FPR", "FNR", "Recall", "Precision", "Accuracy"
        ]])
        sns.boxplot(x="variable", y="value", data=df_merge)
        plt.grid(axis='x')
        plt.savefig(f"occupied_metrics.png")
        plt.close()"""
    return results
