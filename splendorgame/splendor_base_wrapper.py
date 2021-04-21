from collections import defaultdict
from typing import Optional

import gym
import numpy as np
from splendor.core.core import Color
from splendor.core.splendor import Action, ActionType


def all_pick_action(token_num = 3):
    """
        得到选取筹码动作 pick 的全部可能集合
        [<Color.green: 1>, <Color.white: 2>, <Color.blue: 3>, <Color.black: 4>, <Color.red: 5>, <Color.yellow: 6>]
        [0,1,0,0,1] 表示一个白色，一个红色（pick不可以取到金色筹码）
        :return:
            pick_acts: (30,5)
    """
    pick_acts = []
    tocken = np.eye(5)
    if token_num>=1:
        for i in range(5):  # 1 个
            pick_acts.append(tocken[i].tolist())
    if token_num>=2:
        for i in range(5):  # 2 个
            for j in range(i, 5):
                pick_acts.append((tocken[i] + tocken[j]).tolist())
    if token_num>=3:
        for i in range(3):
            for j in range(i + 1, 4):
                for k in range(j + 1, 5):
                    pick_acts.append((tocken[i] + tocken[j] + tocken[k]).tolist())
    return np.array(pick_acts, dtype=np.int)


def good_pick_action(tokens_avail, token_num = 3):
    """
        得到选取筹码动作 pick 的全部可能集合
        [<Color.green: 1>, <Color.white: 2>, <Color.blue: 3>, <Color.black: 4>, <Color.red: 5>, <Color.yellow: 6>]
        [0,1,0,0,1] 表示一个白色，一个红色（pick不可以取到金色筹码）
        :return:
            pick_acts: (30,5)
    """
    pick_acts = []
    tocken = np.eye(5)
    if token_num==0:
        return all_pick_action()
        
    if len(tokens_avail[tokens_avail!=0]) >= 3:
        for i in range(3):
            for j in range(i + 1, 4):
                for k in range(j + 1, 5):
                    pick_acts.append((tocken[i] + tocken[j] + tocken[k]).tolist())
        for i in range(5):
           pick_acts.append((tocken[i]*2).tolist())
    if token_num==2 or len(tokens_avail[tokens_avail!=0]) == 2:
        for i in range(5):  # 2 个
            for j in range(i+1, 5):
                pick_acts.append((tocken[i] + tocken[j]).tolist())
    if token_num==1 or len(tokens_avail[tokens_avail!=0]) == 1:
        for i in range(5):  # 1 个
                pick_acts.append(tocken[i].tolist())
    return np.array(pick_acts, dtype=np.int)

class ActionTransTool:
    """
        包含方法:
        : all_pick_action:
            # return: 所有可取筹码
        : action_map:
            # inp: raw_obs, action编号
            # return: map到环境可用动作
        : valid_action:
            # inp: raw_obs
            # return: valid action mask
    """

    pick_actions = all_pick_action()
    color_map = [color for color in Color if color != Color.yellow]

    @classmethod
    def action_map(cls, obs, action):
        """
        按动作序号解析为对应环境动作：
            0～29:    对应可以pick的筹码组合,共30种
            30～41:   对应台面上可以buy的三个阶级所有卡牌，共3*4=12种
            42～44:   对应锁定的可以buy的卡牌，共3种
            45～56:   对应台面上可以reserve的三个阶级的所有卡牌，共3*4=12种
            57～61:   对应手上可以return的筹码，共5种
            62:       空过
        """
        player = obs.players[obs.current_player]
        cards_index = np.zeros((3, 4), dtype=np.int)
        if action < 30:  # pick
            act = Action(obs.current_player, ActionType.pick)
            for token_idx in range(len(cls.pick_actions[action])):
                act.info[cls.color_map[token_idx]] = int(
                    cls.pick_actions[action][token_idx]
                )

        elif action < 30 + 12:  # buy
            tiers = [obs.cards.tier1, obs.cards.tier2, obs.cards.tier3]
            card_index = tiers[(action - 30) // 4][(action - 30) % 4].index
            act = Action(
                index=obs.current_player, typ=ActionType.buy, info=int(card_index)
            )
        elif action < 30 + 12 + 3:  # buy reserved
            card_index = player.reserved_cards_index[action - 30 - 12]

            act = Action(
                 index=obs.current_player, typ=ActionType.buy, info=int(card_index)
            )

        elif action < 30 + 12 + 3 + 12:  # reserve
            tiers = [obs.cards.tier1, obs.cards.tier2, obs.cards.tier3]
            card_index = tiers[(action - 30 - 12 - 3) // 4][
                (action - 30 - 12 - 3) % 4
            ].index
            act = Action(
                index=obs.current_player, typ=ActionType.reserve, info=int(card_index)
            )

        elif action < 30 + 12 + 3 + 12 + 5:  # return
            act = Action(index=obs.current_player, typ=ActionType.returns)
            act.info = defaultdict(int)
            act.info[cls.color_map[action - 30 - 12 - 3 - 12]] = 1

        elif action == 62:  # pass
            act = Action(index=obs.current_player, typ=ActionType.pick)
            act.info = defaultdict(int)
        return act

    @classmethod
    def valid_action(cls, obs):
        """
            输入为环境原始obs，返回对应位置的定义出的action mask,其中:
            pick_mask:      拿取筹码的方式，总计30种拿取方式    (30,)
            buy_mask:       买台面上卡片，总计12种选取方式      (12,)
            buy_reserved_mask: 购买已经锁定的卡片，总计3种选取方式 (3,)
            reserve_mask:   锁定台面上的卡牌，总计12种选取方式   (12,)
            ret_mask:       归还筹码，默认不归还金色筹码，总计5种方式   (5,)
            pass_mask:      空过，当需要归还筹码时为False,其余时刻均为True  (1,)
            : return:
                mask:   返回拼接后的mask信息    (63,)

        """
        player = obs.players[obs.current_player]
        pick_mask = np.zeros(len(cls.pick_actions), dtype=np.bool)
        buy_mask = np.zeros((3, 4), dtype=np.bool)
        buy_reserved_mask = np.zeros(3, dtype=np.bool)
        reserve_mask = np.zeros((3, 4), dtype=np.bool)
        ret_mask = np.zeros(5, dtype=np.bool)
        pass_mask = np.zeros(1, dtype=np.bool)

        player_token = player.tokens
        # valid_pick_actions = all_pick_action(10-sum(player.tokens.values())) #限定拿完筹码不超过10个，避免退筹码。
        if not obs.need_return:
            # pick tockens valid action
            tokens = np.array(
                [obs.tokens[color] for color in Color if color != Color.yellow]
            )
            good_actions = good_pick_action(tokens, 10-sum(player.tokens.values()))
            for i in range(len(pick_mask)):
                if (
                    (tokens < cls.pick_actions[i]).any()
                    or (2 in cls.pick_actions[i] and tokens[np.where(cls.pick_actions[i] == 2)[0]] < 4)
                    # or cls.pick_actions[i] not in valid_pick_actions
                ):
                    continue
                for good_action in good_actions:
                    if (cls.pick_actions[i]==good_action).all():
                        pick_mask[i] = True
                        break

            # buy cards valid action and reserve card valid action
            tiers = [obs.cards.tier1, obs.cards.tier2, obs.cards.tier3]
            for tier_idx in range(len(tiers)):
                for card_idx in range(len(tiers[tier_idx])):
                    token_diff = [
                        player_token[color]
                        + player.cards[color]
                        - tiers[tier_idx][card_idx].price[color]
                        for color in Color.enums(True)
                    ]
                    missing_tokens = np.abs(
                        np.sum([num for num in token_diff if num < 0])
                    )

                    if missing_tokens <= player_token[Color.yellow]:
                        buy_mask[tier_idx][card_idx] = True

                    if (
                        obs.tokens[Color.yellow] != 0
                        and len(player.reservations) < 3
                        # and sum(player.tokens.values()) < 10 # 限定必须筹码小于10个才能预定，避免退筹码
                    ):
                        reserve_mask[tier_idx][card_idx] = True
            for idx in range(len(player.reservations)):
                token_diff = [
                    player_token[color]
                    + player.cards[color]
                    - player.reservations[idx].price[color]
                    for color in Color.enums(True)
                ]
                missing_tokens = abs(sum(num for num in token_diff if num < 0))

                if missing_tokens <= player_token[Color.yellow]:
                    buy_reserved_mask[idx] = True

        else:
            # return valid action
            ret_mask = np.array(
                [player_token[color] != 0 for color in Color if color != Color.yellow]
            )
            pass_mask[0] = False

        buy_mask = buy_mask.flatten()
        reserve_mask = reserve_mask.flatten()

        mask = np.concatenate(
            [pick_mask, buy_mask, buy_reserved_mask, reserve_mask, ret_mask, pass_mask]
        )
        if not mask.any():
            mask[-1]=True
        return mask


class ObservationTransTool:
    """
        对raw_obs的可用封装，包含方法：
        : deal_card_info:
            # inp: raw_obs
            # return: 卡牌相关信息向量
        : deal_player_info:
            # inp: raw_obs
            # return: 玩家相关信息向量
        : deal_noble_info:
            # inp: raw_obs
            # return: 贵族相关信息向量
        : deal_other_info
            # inp: raw_obs
            # return: 其他环境相关信息向量
    """

    @classmethod
    def obs_to_state(cls, obs):
        """
            解析所有obs信息，其中：
            : Card_state:   台面所有可见卡牌状态信息（包括已经锁定的）  (168,)
            : Player_state: 所有玩家的状态信息      (52,)
            : Nobles_state: 台面上的贵族状态信息    (25,)
            : Other_state:  其他状态信息    (14,)
            : return:
                game_state  (259, )

        """
        Card_state = cls.deal_card_info(obs)
        Player_state = cls.deal_player_info(obs)
        Nobles_state = cls.deal_noble_info(obs)
        Other_state = cls.deal_other_info(obs)

        game_state = np.concatenate(
            [Card_state, Player_state, Nobles_state, Other_state]
        )
        return game_state

    @staticmethod
    def deal_card_info(obs):
        """
            台面卡牌以及所有锁定卡牌信息，共12+12=24张卡牌
            每张卡牌包含:
                所需宝石数 *5
                卡牌分数 *1
                卡牌颜色 *1
            0～83:      台面卡牌信息
            84～167:    所有玩家锁定的卡牌信息
            return:
                :cards_info: 台面卡牌12张，所有锁定卡牌数12张， 返回信息24*7=168
        """
        player = obs.players[obs.current_player]
        # 台面卡牌
        tiers = [obs.cards.tier1, obs.cards.tier2, obs.cards.tier3]
        cards_info = []
        card_infos = []
        for tier in tiers:
            tier_info = []
            for card in tier:
                card_info = [
                    card.price[color] for color in Color if color != Color.yellow
                ]
                card_info.append(card.color.value)
                card_info.append(card.score)
                tier_info.extend(card_info)
            for _ in range(4 - len(tier)):
                tier_info.extend([-1] * 7)
            cards_info.extend(tier_info)
        # 锁定卡牌

        idxs = [i for i in range(obs.player_num)]
        for _ in range(obs.current_player):
            idxs = idxs[1:]+idxs[:1]
        for idx in idxs:
            player = obs.players[idx]
            reserves_info = []
            for reserve in player.reservations:
                reserves_info.extend(
                    [reserve.price[color] for color in Color if color != Color.yellow]
                )
                reserves_info.append(reserve.color.value)
                reserves_info.append(reserve.score)
            for _ in range(3 - len(player.reservations)):
                reserves_info.extend([-1] * 7)
            cards_info.extend(reserves_info)

            
        cards_info.extend([-1] * 7 * 3 * (4 - len(obs.players)))  # 补全 4 位玩家
        return cards_info

    @staticmethod
    def deal_player_info(obs):
        """
        解析所有玩家信息,包括:
            手中各颜色卡牌数 *5
            手中贵族个数 *1
            玩家分数 *1
            手中个颜色筹码数量 *6
        :obs: 原始环境信息
        return:
            :player_info: 长度为 13*4 = 52
        """

        def _get_info(player_obs):
            # 每一位玩家信息包含 每种颜色的卡牌数量*5， 贵族个数 *1，玩家分数 *1， 筹码数量 *6，共13
            player_info = [player_obs.nobles, player_obs.score]
            cards = list(player_obs.cards.values())  # 5
            tokens = list(player_obs.tokens.values())  # 6

            player_info.extend(cards)
            player_info.extend(tokens)
            return player_info  # 13
        players_info = []  # 103 * obs.player_num
        idx = obs.current_player
        player = obs.players[idx]
        info = _get_info(player)
        players_info.extend(info)
        idx = (idx + 1) % obs.player_num
        while idx != obs.current_player:
            player = obs.players[idx]
            info = _get_info(player)
            players_info.extend(info)
            idx = (idx + 1) % obs.player_num

        players_info.extend([-1 for _ in range(len(info) * (4 - obs.player_num))])
        return players_info

    @staticmethod
    def deal_noble_info(obs):
        """ 台面贵族信息, 每个贵族包含:
            所需颜色*5
            return:
                nobles_info: 5*5=25
        """
        nobles_info = []
        for noble in obs.nobles:  # 5*5   # 最大值为5
            nobles_info.extend(
                [noble.price[color] for color in Color if color != Color.yellow]
            )
        for _ in range(5 - len(obs.nobles)):
            nobles_info.extend([-1] * 5)
        return nobles_info

    @staticmethod
    def deal_other_info(obs):
        """
            处理其他信息其他信息包括：
            当前玩家序号 *4 one-hot
            当前回合 *1
            各阶还未出现的卡牌数 *3
            台面各颜色筹码数 *6
            return:
                other_info: 4+3+1+6 = 14
        """
        other_info = []
        current_player = np.eye(4)[obs.current_player]  # 3
        play_round = [obs.play_round]  # 1
        hidden_cards_num = [
            obs.hidden_cards.tier1,
            obs.hidden_cards.tier2,
            obs.hidden_cards.tier3,
        ]  # 3

        tokens = [obs.tokens[color] for color in Color]  # 6

        other_info.extend(current_player)
        other_info.extend(play_round)
        other_info.extend(hidden_cards_num)
        other_info.extend(tokens)
        return other_info


class SplendorBaseWrapper(gym.Wrapper):
    """
        宝石商人原始env的wrapper
    """

    def __init__(self, env):
        super().__init__(env)
        obs_space = gym.spaces.Box(
            low=0, high=100, shape=(259,), dtype=int
        )
        self.action_space = gym.spaces.Discrete(63)
        self.observation_space = gym.spaces.Dict(dict(obs=obs_space,mask=gym.spaces.Box(low=0,high=1,shape=(self.action_space.n,),dtype=np.bool)))

    def step(self, action):
        if isinstance(action,list):
            action = action[0]
        obs = self.raw_observation()
        act = ActionTransTool.action_map(obs, action)
        obs, reward, done, info = self.env.step(act)
        state = ObservationTransTool.obs_to_state(obs)
        # action_mask = ActionTransTool.valid_action(obs)
        self.current_player = self.env.splendor.state.current_player
        return state, reward, done, info

    def reset(self):
        # 随机初始化一个玩家位置
        obs = self.env.reset()
        state = ObservationTransTool.obs_to_state(obs)
        action_mask = ActionTransTool.valid_action(obs)
        self.last_score = 0
        self.current_player = self.env.splendor.state.current_player
        def sample():
            return np.random.choice(np.where(np.ones(shape=(self.action_space.n)) * action_mask==1)[0])
        self.action_space.sample = sample
        return state
    
    def seed(self,seed):
        self.env.seed(seed)

    def raw_observation(self):
        return self.env.splendor.observation()