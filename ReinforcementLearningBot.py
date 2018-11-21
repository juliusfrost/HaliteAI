"""
import all necessary libraries
load the policy

init
for each step in the game:
    update the frame
    get input to feed into policy gradient
    select action for player
    for each ship:
        get ship specific input
        select action for ship

"""

#!/usr/bin/env python3
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.game_map import MapCell
from hlt.positionals import Direction
from hlt.positionals import Position

# This library allows you to generate random numbers.
import random

# ml libraries
import tensorflow as tf
import trfl
import numpy as np

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("MyPythonBot")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))



ship_state = dict()


""" <<<Game Loop>>> """

while True:
    # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
    #   running update_frame().
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map


    height = game_map.height
    width = game_map.height

    halite_grid = [[game_map[Position(row, col)].halite_amount for col in range(width)] for row in range(height)]
    #ship_grid = [[game_map[Position(row, col)].halite_amount for col in range(width)] for row in range(height)]

    # player specific information
    # game.players is a dictionary of form {player_id: Player object}
    current_turn = game.turn_number
    players_halite = [player.halite_amount for player in game.players.values()]
    players_shipyard = [player.shipyard() for player in game.players.values()]
    players_dropoffs = [player.get_dropoffs() for player in game.players.values()]
    players_ships = [player.get_ships() for player in game.players.values()]

    # generate numpy arrays for ml processing
    players_ships_positions = np.zeros((len(game.players), height, width))
    players_ships_halite = np.zeros((len(game.players), height, width))
    players_dropoffs_grid = np.zeros((len(game.players), height, width))
    players_shipyard_grid = np.zeros((len(game.players), height, width))

    for player_id in range(len(game.players)):
        for ship in players_ships[player_id]:
            position = ship.position
            x = position.x
            y = position.y
            players_ships_positions[player_id, y, x] = 1
            players_ships_halite[player_id, y, x] = ship.halite_amount
        for dropoff in players_dropoffs[player_id]:
            position = dropoff.position
            x = position.x
            y = position.y
            players_dropoffs_grid[player_id, y, x] = 1
        shipyard = players_shipyard[player_id]
        position = shipyard.position
        x = position.x
        y = position.y
        players_shipyard_grid[player_id, y, x] = 1

    concat = np.concatenate((players_ships_positions,
                             players_ships_halite,
                             players_dropoffs_grid,
                             players_shipyard_grid,
                             np.reshape(np.array(halite_grid), (1, height, width))),
                            axis = 0)











    # Order the ships by the most halite first
    ship_order = [(ship.halite_amount, ship) for ship in me.get_ships()]
    ship_order.sort(key = lambda x: x[0], reverse=True)
    ship_iter = [x[1] for x in ship_order]



    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []

    for ship in ship_iter:
        # update the ship's state

        if game_map[ship.position].halite_amount + ship.halite_amount > (constants.DROPOFF_COST + constants.SHIP_COST)*1.2:
            command_queue.append(ship.make_dropoff())
            continue

        if not ship.id in ship_state.keys():
            ship_state[ship.id] = {'outbound' : True}

        if ship_state[ship.id]['outbound']:
            if ship.halite_amount > constants.MAX_HALITE  * 0.9:
                ship_state[ship.id]['outbound'] = False
            else:
                ship_state[ship.id]['outbound'] = True
        else:
            if ship.halite_amount > constants.MAX_HALITE * 0.7:
                ship_state[ship.id]['outbound'] = False
            else:
                ship_state[ship.id]['outbound'] = True


        direction = Direction.Still

        if ship_state[ship.id]['outbound']:
            max_halite = game_map[ship.position].halite_amount
            direction = Direction.Still
            for position in ship.position.get_surrounding_cardinals():
                map_cell = game_map[position]
                assert isinstance(map_cell, MapCell)
                h = map_cell.halite_amount
                if h > max_halite and not map_cell.is_occupied:
                    max_halite = h
                    diff = position - ship.position
                    direction = (diff.x, diff.y)

        else:
            direction = game_map.naive_navigate(ship, me.shipyard.position)

        logging.info(str(ship) + ': ' + str(direction))
        position_next = ship.position.directional_offset(direction)
        game_map[position_next].mark_unsafe(ship)
        if direction == Direction.Still:
            command_queue.append(ship.stay_still())
        else:
            command_queue.append(ship.move(direction))




        #choices = ship.position.get_surrounding_cardinals()



        # For each of your ships, move randomly if the ship is on a low halite location or the ship is full.
        #   Else, collect halite.
        #if game_map[ship.position].halite_amount < constants.MAX_HALITE / 10 or ship.is_full:
        #    command_queue.append(
        #        ship.move(
        #            random.choice([ Direction.North, Direction.South, Direction.East, Direction.West ])))
        #else:
        #    command_queue.append(ship.stay_still())

    # If the game is in the first 200 turns and you have enough halite, spawn a ship.
    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)

