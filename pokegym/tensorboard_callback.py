from stable_baselines3.common.callbacks import BaseCallback


def merge_dicts_by_mean(dicts):
    sum_dict = {}
    count_dict = {}

    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)):
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1

    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]

    return mean_dict


class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:

        if self.training_env.env_method("get_check_if_done", indices=[0])[0]:
            all_infos = self.training_env.get_attr("agent_stats")
            all_final_infos = [stats[-1] for stats in all_infos]
            mean_infos = merge_dicts_by_mean(all_final_infos)
            for key, val in mean_infos.items():
                self.logger.record(f"env_stats/{key}", val)

        return True