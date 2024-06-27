from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import sys
import os
mammoth_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
from datasets.seq_nturgbd60 import Ntu60

actions = {
    1: "drink water",
    2: "eat meal/snack",
    3: "brushing teeth",
    4: "brushing hair",
    5: "drop",
    6: "pickup",
    7: "throw",
    8: "sitting down",
    9: "standing up (from sitting position)",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "wear jacket",
    15: "take off jacket",
    16: "wear a shoe",
    17: "take off a shoe",
    18: "wear on glasses",
    19: "take off glasses",
    20: "put on a hat/cap",
    21: "take off a hat/cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something",
    25: "reach into pocket",
    26: "hopping (one foot jumping)",
    27: "jump up",
    28: "make a phone call/answer phone",
    29: "playing with phone/tablet",
    30: "typing on a keyboard",
    31: "pointing to something with finger",
    32: "taking a selfie",
    33: "check time (from watch)",
    34: "rub two hands together",
    35: "nod head/bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put the palms together",
    40: "cross hands in front (say stop)",
    41: "sneeze/cough",
    42: "staggering",
    43: "falling",
    44: "touch head (headache)",
    45: "touch chest (stomachache/heart pain)",
    46: "touch back (backache)",
    47: "touch neck (neckache)",
    48: "nausea or vomiting condition",
    49: "use a fan (with hand or paper)/feeling warm",
    50: "punching/slapping other person",
    51: "kicking other person",
    52: "pushing other person",
    53: "pat on back of other person",
    54: "point finger at the other person",
    55: "hugging other person",
    56: "giving something to other person",
    57: "touch other person's pocket",
    58: "handshaking",
    59: "walking towards each other",
    60: "walking apart from each other",
    61: "put on headphone",
    62: "take off headphone",
    63: "shoot at the basket",
    64: "bounce ball",
    65: "tennis bat swing",
    66: "juggling table tennis balls",
    67: "hush (quite)",
    68: "flick hair",
    69: "thumb up",
    70: "thumb down",
    71: "make ok sign",
    72: "make victory sign",
    73: "staple book",
    74: "counting money",
    75: "cutting nails",
    76: "cutting paper (using scissors)",
    77: "snapping fingers",
    78: "open bottle",
    79: "sniff (smell)",
    80: "squat down",
    81: "toss a coin",
    82: "fold paper",
    83: "ball up paper",
    84: "play magic cube",
    85: "apply cream on face",
    86: "apply cream on hand back",
    87: "put on bag",
    88: "take off bag",
    89: "put something into a bag",
    90: "take something out of a bag",
    91: "open a box",
    92: "move heavy objects",
    93: "shake fist",
    94: "throw up cap/hat",
    95: "hands up (both hands)",
    96: "cross arms",
    97: "arm circles",
    98: "arm swings",
    99: "running on the spot",
    100: "butt kicks (kick backward)",
    101: "cross toe touch",
    102: "side kick",
    103: "yawn",
    104: "stretch oneself",
    105: "blow nose",
    106: "hit other person with something",
    107: "wield knife towards other person",
    108: "knock over other person (hit with body)",
    109: "grab other person stuff",
    110: "shoot at other person with a gun",
    111: "step on foot",
    112: "high-five",
    113: "cheers and drink",
    114: "carry something with other person",
    115: "take a photo of other person",
    116: "follow other person",
    117: "whisper in other person ear",
    118: "exchange things with other person",
    119: "support somebody with hand",
    120: "finger-guessing game (playing rock-paper-scissors)",
}

ntu_skeleton_bone_pairs = tuple((i - 1, j - 1) for (i, j) in (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
))


def visualize_pair(skeleton: torch.Tensor, label: torch.Tensor = None):
    """generate gif of the skeleton with two people

    Args:
        skeleton (torch.Tensor): unbatched skeleton (C, T, V, M)
        label (torch.Tensor, optional): Defaults to None.
    """
    bones = ntu_skeleton_bone_pairs

    mpl.rcParams['legend.fontsize'] = 10
    fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection='3d'))
    for ax in axes:
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.view_init(azim=-90, elev=90)
        ax.set_axis_off()

    def animate(skeleton):
        for n, ax in enumerate(axes):
            ax.clear()
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_axis_off()

            sk = skeleton[..., n]
            for i, j in bones:
                joint_locs = sk[:, [i, j]]
                # plot them
                ax.plot(joint_locs[0], joint_locs[1], joint_locs[2], color='blue')

            skeleton_index[0] += 1
        if label is not None:
            fig.suptitle(actions[label.item() + 1], fontsize=16)

        plt.tight_layout()

    skeleton_index = [0]
    skeleton_frames = skeleton.permute(1, 0, 2, 3).numpy()
    ani = FuncAnimation(fig, animate, skeleton_frames)
    ani.save('skeleton.gif', writer='pillow', fps=30)


def visualize_single(skeleton: torch.Tensor, label: torch.Tensor = None):
    """generate gif of the skeleton with one person

    Args:
        skeleton (torch.Tensor): unbatched skeleton (C, T, V, M)
        label (torch.Tensor, optional): Defaults to None.
    """

    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(azim=-90, elev=90)
    ax.set_axis_off()

    def animate(skeleton):
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_axis_off()

        for i, j in ntu_skeleton_bone_pairs:
            joint_locs = skeleton[:, [i, j]]
            # plot them
            ax.plot(joint_locs[0], joint_locs[1], joint_locs[2], color='blue')

    skeleton_frames = skeleton.permute(1, 0, 2, 3).numpy()
    ani = FuncAnimation(fig, animate, skeleton_frames)
    ani.save('skeleton.gif', writer='pillow', fps=30)


def visualize_only_joints(skeleton: torch.Tensor, label: torch.Tensor = None):
    """generate single images of the skeleton (not gif)

    Args:
        skeleton (torch.Tensor): unbatched skeleton (C, T, V, M)
        label (torch.Tensor, optional): Defaults to None.
    """
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(azim=-90, elev=90)
    ax.set_axis_off()
    skeleton = skeleton.transpose(1, 0, 2, 3)
    Path.mkdir(Path('sium'), exist_ok=True)
    for n, frame in enumerate(skeleton):
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_axis_off()

        for i, j in ntu_skeleton_bone_pairs:
            joint_locs = frame[:, [i, j]][:, :, 0]
            # plot them
            ax.plot(joint_locs[0], joint_locs[1], joint_locs[2], color='blue', solid_capstyle='round')

        fig.savefig('sium/skeleton' + str(n) + '.svg', format='svg', dpi=1000, transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    ds = Ntu60('./data/NTU60_XView.npz', split="test",
               p_interval=[0.95], window_size=120, permute=0)

    inp = ds._data_final[ds.targets == 23][0]
    two = inp[..., 1].sum((0, 1, 2)) > 0

    visualize_only_joints(inp)
