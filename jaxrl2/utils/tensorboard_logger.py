import os
import datetime

import dateutil.tz
from collections import OrderedDict
import numpy as np
from numbers import Number
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import imageio
from torch.utils.tensorboard import SummaryWriter


def create_exp_name(exp_prefix, exp_id=0, seed=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return "%s_%s_%04d--s-%d" % (exp_prefix, timestamp, exp_id, seed)


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})
    try:
        stats = OrderedDict([
            (name + ' Mean', np.mean(data)),
            (name + ' Std', np.std(data)),
        ])
    except:
        stats = OrderedDict([
            (name + ' Mean', -1),
            (name + ' Std', -1),
        ])
    if not exclude_max_min:
        try:
            stats[name + ' Max'] = np.max(data)
            stats[name + ' Min'] = np.min(data)
        except:
            stats[name + ' Max'] = -1
            stats[name + ' Min'] = -1
    return stats


def _to_hwc_uint8(frame):
    """Coerce a single image/frame to (H, W, C) uint8 for imageio/TensorBoard."""
    arr = np.asarray(frame)
    # (C, H, W) -> (H, W, C) when the leading axis is the channel axis.
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


class Video(object):
    """Drop-in replacement for ``wandb.Video``.

    Holds a video as a numpy array; ``TensorBoardLogger.log`` materialises it to
    disk (mp4 by default, gif on request). Accepts ``(T, C, H, W)`` or
    ``(T, H, W, C)`` frames, matching the layout ``wandb.Video`` accepted.
    """

    def __init__(self, data, fps=30, format="mp4"):
        self.data = np.asarray(data)
        self.fps = int(fps)
        self.format = format


class Image(object):
    """Drop-in replacement for ``wandb.Image`` (numpy-array image)."""

    def __init__(self, data, caption=None):
        self.data = np.asarray(data)
        self.caption = caption


class TensorBoardLogger(object):
    """Logging shim with the same interface the old ``WandBLogger`` exposed.

    - ``log(data: dict, step=None)``: scalar values go to ``add_scalar``;
      :class:`Video` / :class:`Image` values are written to local ``videos/`` /
      ``images/`` subdirectories of the run directory (images are also mirrored
      into TensorBoard).
    - ``log_histogram(name, values, step)``: native TensorBoard histogram.

    ``output_dir`` is used directly as the TensorBoard log dir (the run
    directory under ``$EXP``), so events and saved media live next to the run.
    """

    def __init__(self, logging_enabled, variant, project, experiment_id,
                 output_dir=None, group_name='', team=None):
        self.logging_enabled = bool(logging_enabled)
        if output_dir is None:
            output_dir = os.path.join('logs', experiment_id)
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.experiment_id = experiment_id
        self.video_dir = os.path.join(output_dir, 'videos')
        self.image_dir = os.path.join(output_dir, 'images')
        self.writer = None

        if self.logging_enabled:
            os.makedirs(self.video_dir, exist_ok=True)
            os.makedirs(self.image_dir, exist_ok=True)
            print('tensorboard logging to:', output_dir)
            print('tensorboard project:', project)
            print('tensorboard group:', group_name)
            self.writer = SummaryWriter(log_dir=output_dir)
            try:
                config_items = dict(variant)
            except (TypeError, ValueError):
                config_items = {'variant': str(variant)}
            config_text = '\n'.join(
                ['project: {}'.format(project), 'group: {}'.format(group_name)]
                + ['{}: {}'.format(k, v) for k, v in config_items.items()]
            )
            self.writer.add_text('config', config_text, 0)

    @staticmethod
    def _coerce_step(step):
        if step is None:
            return None
        try:
            return int(step)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_name(key):
        return (str(key).replace('/', '_').replace(' ', '_')
                .replace('>=', 'ge').replace('=', '').replace('>', 'gt'))

    def log(self, data, step=None, **kwargs):
        """Log a dict of values. ``step`` may be passed positionally or by name,
        matching the old ``wandb.log(data, step=...)`` call sites."""
        if not self.logging_enabled or self.writer is None:
            return
        if not isinstance(data, dict):
            return
        step = self._coerce_step(step)
        for key, value in data.items():
            if isinstance(value, Video):
                self._log_video(key, value, step)
            elif isinstance(value, Image):
                self._log_image(key, value, step)
            elif isinstance(value, Number):
                self.writer.add_scalar(key, float(value), global_step=step)
            elif isinstance(value, np.ndarray) and value.ndim == 0:
                self.writer.add_scalar(key, float(value), global_step=step)
            else:
                try:
                    self.writer.add_scalar(key, float(value), global_step=step)
                except (TypeError, ValueError):
                    pass
        self.writer.flush()

    def _log_video(self, key, video, step):
        frames = [_to_hwc_uint8(f) for f in video.data]
        stem = self._safe_name(key)
        step_tag = '' if step is None else '_step{}'.format(step)
        ext = 'gif' if video.format == 'gif' else 'mp4'
        path = os.path.join(self.video_dir, '{}{}.{}'.format(stem, step_tag, ext))
        try:
            if ext == 'mp4':
                imageio.mimwrite(path, frames, fps=video.fps, macro_block_size=None)
            else:
                imageio.mimwrite(path, frames, fps=video.fps)
        except Exception as exc:  # noqa: BLE001 - fall back to gif if mp4 backend missing
            try:
                path = os.path.join(self.video_dir,
                                    '{}{}.gif'.format(stem, step_tag))
                imageio.mimwrite(path, frames, fps=video.fps)
            except Exception as exc2:  # noqa: BLE001
                print('[TensorBoardLogger] failed to write video {}: {}'.format(path, exc2))
                return
        # Record where the file lives so it is discoverable from TensorBoard.
        self.writer.add_text(key, os.path.relpath(path, self.output_dir), global_step=step)
        # Also embed an animated GIF summary so the video plays inside the
        # TensorBoard IMAGES tab (frames subsampled to bound event-file size).
        try:
            self._embed_gif_summary(key, frames, video.fps, step)
        except Exception as exc:  # noqa: BLE001 - embedding is best-effort
            print('[TensorBoardLogger] failed to embed video {}: {}'.format(key, exc))
        self.writer.flush()

    _MAX_EMBED_FRAMES = 80

    def _embed_gif_summary(self, key, frames, fps, step):
        import io
        from tensorboard.compat.proto.summary_pb2 import Summary

        stride = max(1, int(np.ceil(len(frames) / float(self._MAX_EMBED_FRAMES))))
        sub = frames[::stride]
        sub_fps = max(1, int(round(fps / float(stride))))
        buf = io.BytesIO()
        try:
            imageio.mimwrite(buf, sub, format='gif', fps=sub_fps, loop=0)
        except TypeError:  # newer imageio versions use duration (ms) instead of fps
            buf = io.BytesIO()
            imageio.mimwrite(buf, sub, format='gif', duration=1000.0 / sub_fps, loop=0)
        h, w = sub[0].shape[:2]
        image = Summary.Image(height=h, width=w, colorspace=3,
                              encoded_image_string=buf.getvalue())
        summary = Summary(value=[Summary.Value(tag=key, image=image)])
        self.writer._get_file_writer().add_summary(summary, global_step=step)

    def _log_image(self, key, image, step):
        arr = _to_hwc_uint8(image.data)
        stem = self._safe_name(key)
        step_tag = '' if step is None else '_step{}'.format(step)
        path = os.path.join(self.image_dir, '{}{}.png'.format(stem, step_tag))
        try:
            imageio.imwrite(path, arr)
        except Exception as exc:  # noqa: BLE001
            print('[TensorBoardLogger] failed to write image {}: {}'.format(path, exc))
        try:
            self.writer.add_image(key, arr, global_step=step, dataformats='HWC')
        except Exception:  # noqa: BLE001
            pass
        self.writer.flush()

    def log_histogram(self, name, values, step):
        if not self.logging_enabled or self.writer is None:
            return
        values = np.asarray(values).flatten()
        step = self._coerce_step(step)
        try:
            self.writer.add_histogram(name, values, global_step=step)
        except Exception:  # noqa: BLE001 - degenerate inputs fall back to a scalar
            try:
                self.writer.add_scalar(name + '_mean', float(np.mean(values)),
                                       global_step=step)
            except Exception:  # noqa: BLE001
                pass
        self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
