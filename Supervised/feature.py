import numpy as np
from agent import MahjongGBAgent
from collections import defaultdict

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise


class FeatureAgent(MahjongGBAgent):
    """
    observation: 6*4*9
        4 * 9: Represent 32 tiles in one-hot encoding.
        133 channels:
            - hand 4
            - (chi 4 + peng 1 + gang 1) * 4 players
            - angang 1
            - play history 25 * 4 players
            - unknown tiles 4
    action_mask: 235
        pass1+hu1+discard34+chi63(3*7*3)+peng34+gang34+angang34+bugang34
    """

    OBS_SIZE = 133
    ACT_SIZE = 235

    OFFSET_OBS = {
        'HAND': 0,
        'CHI_0': 4,
        'CHI_1': 8,
        'CHI_2': 12,
        'CHI_3': 16,
        'PENG_0': 20,
        'PENG_1': 21,
        'PENG_2': 22,
        'PENG_3': 23,
        'GANG_0': 24,
        'GANG_1': 25,
        'GANG_2': 26,
        'GANG_3': 27,
        'ANGANG': 28,
        'HISTORY_0': 29,
        'HISTORY_1': 54,
        'HISTORY_2': 79,
        'HISTORY_3': 104,
        'UNKNOWN': 129
    }
    OFFSET_ACT = {
        'Pass': 0,
        'Hu': 1,
        'Play': 2,
        'Chi': 36,
        'Peng': 99,
        'Gang': 133,
        'AnGang': 167,
        'BuGang': 201
    }
    TILE_LIST = [
        *('W%d' % (i + 1) for i in range(9)),
        *('T%d' % (i + 1) for i in range(9)),
        *('B%d' % (i + 1) for i in range(9)),
        *('F%d' % (i + 1) for i in range(4)),
        *('J%d' % (i + 1) for i in range(3))
    ]
    OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}

    def __init__(self, seatWind):
        super().__init__(seatWind)
        self.seatWind = seatWind
        self.packs = [[] for i in range(4)]
        self.chi_cnt = [0 for i in range(4)]
        self.history = [[] for i in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.wallLast = False
        self.isAboutKong = False
        self.obs = np.zeros((self.OBS_SIZE, 36))
        self.obs[self.OFFSET_OBS['UNKNOWN']: self.OFFSET_OBS['UNKNOWN'] + 4, :] = 1
        self.known_tile_cnt = {c: 0 for c in self.TILE_LIST}

    '''
    Wind 0..3
    Deal XX XX ...
    Player N Draw
    Player N Gang
    Player N(me) AnGang XX
    Player N(me) Play XX
    Player N(me) BuGang XX
    Player N(not me) Peng
    Player N(not me) Chi XX
    Player N(not me) AnGang
    
    Player N Hu
    Huang
    Player N Invalid
    Draw XX
    Player N(not me) Play XX
    Player N(not me) BuGang XX
    Player N(me) Peng
    Player N(me) Chi XX
    '''

    def request2obs(self, request):
        t = request.split()
        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            return
        if t[0] == 'Deal':
            # Allocate hand tiles.
            self.hand = t[1:]
            self._hand_embedding_update()
            self._update_known_tile(t[1:])
            return
        if t[0] == 'Huang':
            self.valid = []
            return self._obs()
        if t[0] == 'Draw':
            # Draw a tile.
            # Available next step: Hu, Play, AnGang, BuGang
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []
            if self._check_mahjong(tile, isSelfDrawn=True, isAboutKong=self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            # After drawing, we know one more tile.
            self._update_known_tile([tile])
            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile])
            return self._obs()
        # Player N Invalid/Hu/Draw/Play/Chi/Peng/Gang/AnGang/BuGang XX
        p = (int(t[1]) + 4 - self.seatWind) % 4
        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return
        if t[2] == 'Invalid':
            self.valid = []
            return self._obs()
        if t[2] == 'Hu':
            self.valid = []
            return self._obs()
        if t[2] == 'Play':
            # Drop a tile.
            self.tileFrom = p
            self.curTile = t[3]
            self.shownTiles[self.curTile] += 1
            # Update the play history.
            self.obs[self.OFFSET_OBS[f'HISTORY_{p}'] + len(self.history[p])][self.OFFSET_TILE[t[3]]] = 1
            self.history[p].append(self.curTile)

            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                # Available: Hu/Gang/Peng/Chi/Pass
                self._update_known_tile([self.curTile])  # After anather player plays a tile, we know one more tile.
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3): tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        if t[2] == 'Chi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                tile_name = color + str(num + i)
                self.shownTiles[tile_name] += 1
                # Update the feature of "Chi".
                self.obs[self.OFFSET_OBS[f'CHI_{p}'] + self.chi_cnt[p]][self.OFFSET_TILE[tile_name]] = 1
            self.chi_cnt[p] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                # After another player "chi", we know two more tiles.
                newly_known_tiles = []
                for i in range(-1, 2):
                    tile_name = color + str(num + i)
                    if tile_name != self.curTile:
                        newly_known_tiles.append(tile_name)
                self._update_known_tile(newly_known_tiles)
                return
        if t[2] == 'UnChi':
            # Be careful with the priority: Hu > Peng / Gang > Chi
            # TODO: If 'UnChi', will other players see two more tiles for "Chi"?
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            # Update the feature of "Chi".
            self.obs[self.OFFSET_OBS[f'CHI_{p}'] + self.chi_cnt[p], :] = 0
            self.chi_cnt[p] -= 1
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            # Update the feature of "Peng".
            self.obs[self.OFFSET_OBS[f'PENG_{p}'], self.OFFSET_TILE[self.curTile]] = 1
            if p == 0:
                # Available: Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                # When another player 'peng', we know two more tiles.
                self._update_known_tile([self.curTile] * 2)
                return
        if t[2] == 'UnPeng':
            # Be careful with the priority: Hu > Peng / Gang > Chi
            # TODO: If 'UnPeng', will other players see two more tiles for "Peng"?
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            # Update the feature of 'Peng'.
            self.obs[self.OFFSET_OBS[f'PENG_{p}'], self.OFFSET_TILE[self.curTile]] = 0
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            # Update the feature of "Gang".
            self.obs[self.OFFSET_OBS[f'GANG_{p}'], self.OFFSET_TILE[self.curTile]] = 1
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            else:
                # When another player 'gang', we know three more tiles.
                self._update_known_tile([self.curTile] * 3)
            return
        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            if p == 0:
                # After the feature of 'AnGang'.
                self.obs[self.OFFSET_OBS['ANGANG'], self.OFFSET_TILE[tile]] = 1
                self.isAboutKong = True
                for i in range(4):
                    self.hand.remove(tile)
            else:
                self.isAboutKong = False
            return
        if t[2] == 'BuGang':
            # Be careful with the priority: Hu > Peng / Gang > Chi
            # TODO: If 'BuGang', will other players see two more tiles for "Gang"?
            tile = t[3]
            # Update the feature of "Gang".
            self.obs[self.OFFSET_OBS[f'GANG_{p}'], self.OFFSET_TILE[tile]] = 0
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                # Available: Hu/Pass
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn=False, isAboutKong=True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        raise NotImplementedError('Unknown request %s!' % request)

    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    '''

    def action2response(self, action):
        if action < self.OFFSET_ACT['Hu']:
            return 'Pass'
        if action < self.OFFSET_ACT['Play']:
            return 'Hu'
        if action < self.OFFSET_ACT['Chi']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']:
            return 'Peng'
        if action < self.OFFSET_ACT['AnGang']:
            return 'Gang'
        if action < self.OFFSET_ACT['BuGang']:
            return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]

    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    '''

    def response2action(self, response):
        t = response.split()
        if t[0] == 'Pass': return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu': return self.OFFSET_ACT['Hu']
        if t[0] == 'Play': return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi': return self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(
            t[1][1]) - int(t[2][1]) + 1
        if t[0] == 'Peng': return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang': return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang': return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang': return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']

    def _obs(self):
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            mask[a] = 1
        return {
            'observation': self.obs.reshape((self.OBS_SIZE, 4 * 9)).copy(),
            'action_mask': mask
        }

    def _hand_embedding_update(self):
        self.obs[self.OFFSET_OBS['HAND']: self.OFFSET_OBS['HAND'] + 4] = 0
        d = defaultdict(int)
        for tile in self.hand:
            d[tile] += 1
        for tile in d:
            self.obs[self.OFFSET_OBS['HAND']: self.OFFSET_OBS['HAND'] + d[tile], self.OFFSET_TILE[tile]] = 1

    def _update_known_tile(self, newly_known):
        d = defaultdict(int)
        for tile in newly_known:
            d[tile] += 1
        for tile in d:
            self.known_tile_cnt[tile] += d[tile]
            assert self.known_tile_cnt[tile] <= 4, 'Error occurs in known_tile_cnt!'
            offset = self.OFFSET_OBS['UNKNOWN'] + 4
            self.obs[offset - self.known_tile_cnt[tile]: offset, self.OFFSET_TILE[tile]] = 0

    def _check_mahjong(self, winTile, isSelfDrawn=False, isAboutKong=False):
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[0]),
                hand=tuple(self.hand),
                winTile=winTile,
                flowerCount=0,
                isSelfDrawn=isSelfDrawn,
                is4thTile=self.shownTiles[winTile] == 4,
                isAboutKong=isAboutKong,
                isWallLast=self.wallLast,
                seatWind=self.seatWind,
                prevalentWind=self.prevalentWind,
                verbose=True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8: raise Exception('Not Enough Fans')
        except:
            return False
        return True
