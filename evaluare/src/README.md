# Double Double Dominoes Game CLI

## Description
The `game.py` script is a command-line tool for interacting with Double Double Dominoes games. It allows you to process game images, load configurations (board cleaning, extracting dominoes, cleaning dominoes) and test game moves validity.

## Dependencies

To run the script, you need to have the following packages installed:

- python==3.x
- numpy==1.26.2
- opencv-python==4.8.1.78
- tqdm==4.66.1

You can install these dependencies using `pip` with the following command:

```bash
$ pip install -r requirements.txt
```

## Usage
- Game data path should be provided as a command line argument. This path should contain images in the following format: `<game_index>_<move_index>.jpg` and `<game_index>_mutari.txt`.
- Configurations should be stored in the same folder as the script.
    - `board_config.pickle` - board cleaning configuration
    - `extract_domino_config.pickle` - domino extraction configuration
    - `clean_domino_config.pickle` - domino cleaning configuration


### Options for `game.py`
- `-p`, `--game_path` (required): Specifies the path to the game images.
- `-g`, `--game_index` (required): Sets the game index to be processed.
- `-b`, `--board_config`: Path to the board configuration file. Defaults to `board_config.pickle`.
- `-d`, `--extract_domino_config`: Path to the domino extraction configuration file. Defaults to `extract_domino_config.pickle`.
- `-c`, `--clean_domino_config`: Path to the clean domino configuration file. Defaults to `clean_domino_config.pickle`.
- `-t`, `--test`: Runs the game correctness check. This does not require a value.
- `-s`, `--save_game_path`: If specified, saves the game moves as txt to the provided path.

### Examples
To process a game with the minimum required options using default configs:

```bash
$ python game.py -p path/to/game_files/ -g 1
```
---

To process a game with the minimum required options and save the moves to a txt file:

```bash
$ python game.py -p path/to/game_files/ -g 1 -s path/to/save/
```
