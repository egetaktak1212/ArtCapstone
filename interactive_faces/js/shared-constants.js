// Tooltip
const tooltipWidth = 65;
const tooltipHeight = 32;

// const connections = [
//     [[0, 1], [1, 2]],           // left brow
//     [[3, 4], [4, 5]],           // right brow
//     [[6, 7], [7, 8]],           // left eye
//     [[9, 10], [10, 11]],       // right eye
//     [[13, 14], [14, 15], [13, 15], [21, 13], [21, 15]], // nose
//     [[17, 18], [18, 19]]        // mouth
// ]

const connections = [
    [[1, 2], [2, 3]],
    [[4, 5], [5, 6]],
    [[7, 8], [8, 9]],
    [[10, 11], [11, 12]],
    [[14, 15], [15, 16], [14, 16], [22, 14], [22, 16]],
    [[18, 19], [19, 20]]
]

c_names = [
    "l-brow-1", "l-brow-2",
    "r-brow-1", "r-brow-2",
    "l-eye-1", "l-eye-2",
    "r-eye-1", "r-eye-2",
    "nose-1", "nose-2", "nose-3", "bridge-1", "bridge-2",
    "mouth-1", "mouth-2"
]
