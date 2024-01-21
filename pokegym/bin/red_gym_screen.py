import numpy as np
from skimage.transform import resize
import mediapy as media


class RedGymScreen:
    def __init__(self, env):
        if env.debug:
            print('**** RedGymScreen ****')

        self.env = env
        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.frame_stacks = 3
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0],
             self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)
        self.memory_height = 8
        self.recent_memory = np.zeros((self.output_shape[1] * self.memory_height, 3), dtype=np.uint8)
        self.full_frame_writer = None
        self.model_frame_writer= None

        self.prepare_video_recording_if_enabled()

    def render(self, reduce_res=True, update_mem=True):
        game_screen = self.env.gameboy.pyboy.botsupport_manager().screen().screen_ndarray()

        if reduce_res:
            game_screen = self._reduce_resolution(game_screen)

        return game_screen

    def prepare_video_recording_if_enabled(self):
        if self.env.save_video:
            base_dir = self.env.save_path / 'rollouts'
            base_dir.mkdir(exist_ok=True)
            full_name = f'full_reset_{self.env.reset_count}_id{self.env.instance_id}.mp4'
            model_name = f'model_reset_{self.env.reset_count}_id{self.env.instance_id}.mp4'
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.env.output_full[:2], fps=60)
            self.full_frame_writer.__enter__()
            self.model_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False, update_mem=False))
        self.model_frame_writer.add_image(self.render(reduce_res=True, update_mem=False))

    def _reduce_resolution(self, image):
        return (255 * resize(image, self.output_shape)).astype(np.uint8)
