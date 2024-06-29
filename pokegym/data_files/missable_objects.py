# Pad to 32 Bytes
from ctypes import LittleEndianStructure, Union, c_uint8

from pokegym.pyboy_singleton import get_pyboy_instance 

class MissableFlagsBits(LittleEndianStructure):
    _fields_ = [
        ("HS_PALLET_TOWN_OAK", c_uint8, 1),
        ("HS_LYING_OLD_MAN", c_uint8, 1),
        ("HS_OLD_MAN", c_uint8, 1),
        ("HS_MUSEUM_GUY", c_uint8, 1),
        ("HS_GYM_GUY", c_uint8, 1),
        ("HS_CERULEAN_RIVAL", c_uint8, 1),
        ("HS_CERULEAN_ROCKET", c_uint8, 1),
        ("HS_CERULEAN_GUARD_1", c_uint8, 1),
        ("HS_CERULEAN_CAVE_GUY", c_uint8, 1),
        ("HS_CERULEAN_GUARD_2", c_uint8, 1),
        ("HS_SAFFRON_CITY_1", c_uint8, 1),
        ("HS_SAFFRON_CITY_2", c_uint8, 1),
        ("HS_SAFFRON_CITY_3", c_uint8, 1),
        ("HS_SAFFRON_CITY_4", c_uint8, 1),
        ("HS_SAFFRON_CITY_5", c_uint8, 1),
        ("HS_SAFFRON_CITY_6", c_uint8, 1),
        ("HS_SAFFRON_CITY_7", c_uint8, 1),
        ("HS_SAFFRON_CITY_8", c_uint8, 1),
        ("HS_SAFFRON_CITY_9", c_uint8, 1),
        ("HS_SAFFRON_CITY_A", c_uint8, 1),
        ("HS_SAFFRON_CITY_B", c_uint8, 1),
        ("HS_SAFFRON_CITY_C", c_uint8, 1),
        ("HS_SAFFRON_CITY_D", c_uint8, 1),
        ("HS_SAFFRON_CITY_E", c_uint8, 1),
        ("HS_SAFFRON_CITY_F", c_uint8, 1),
        ("HS_ROUTE_2_ITEM_1", c_uint8, 1),
        ("HS_ROUTE_2_ITEM_2", c_uint8, 1),
        ("HS_ROUTE_4_ITEM", c_uint8, 1),
        ("HS_ROUTE_9_ITEM", c_uint8, 1),
        ("HS_ROUTE_12_SNORLAX", c_uint8, 1),
        ("HS_ROUTE_12_ITEM_1", c_uint8, 1),
        ("HS_ROUTE_12_ITEM_2", c_uint8, 1),
        ("HS_ROUTE_15_ITEM", c_uint8, 1),
        ("HS_ROUTE_16_SNORLAX", c_uint8, 1),
        ("HS_ROUTE_22_RIVAL_1", c_uint8, 1),
        ("HS_ROUTE_22_RIVAL_2", c_uint8, 1),
        ("HS_NUGGET_BRIDGE_GUY", c_uint8, 1),
        ("HS_ROUTE_24_ITEM", c_uint8, 1),
        ("HS_ROUTE_25_ITEM", c_uint8, 1),
        ("HS_DAISY_SITTING", c_uint8, 1),
        ("HS_DAISY_WALKING", c_uint8, 1),
        ("HS_TOWN_MAP", c_uint8, 1),
        ("HS_OAKS_LAB_RIVAL", c_uint8, 1),
        ("HS_STARTER_BALL_1", c_uint8, 1),
        ("HS_STARTER_BALL_2", c_uint8, 1),
        ("HS_STARTER_BALL_3", c_uint8, 1),
        ("HS_OAKS_LAB_OAK_1", c_uint8, 1),
        ("HS_POKEDEX_1", c_uint8, 1),
        ("HS_POKEDEX_2", c_uint8, 1),
        ("HS_OAKS_LAB_OAK_2", c_uint8, 1),
        ("HS_VIRIDIAN_GYM_GIOVANNI", c_uint8, 1),
        ("HS_VIRIDIAN_GYM_ITEM", c_uint8, 1),
        ("HS_OLD_AMBER", c_uint8, 1),
        ("HS_CERULEAN_CAVE_1F_ITEM_1", c_uint8, 1),
        ("HS_CERULEAN_CAVE_1F_ITEM_2", c_uint8, 1),
        ("HS_CERULEAN_CAVE_1F_ITEM_3", c_uint8, 1),
        ("HS_POKEMON_TOWER_2F_RIVAL", c_uint8, 1),
        ("HS_POKEMON_TOWER_3F_ITEM", c_uint8, 1),
        ("HS_POKEMON_TOWER_4F_ITEM_1", c_uint8, 1),
        ("HS_POKEMON_TOWER_4F_ITEM_2", c_uint8, 1),
        ("HS_POKEMON_TOWER_4F_ITEM_3", c_uint8, 1),
        ("HS_POKEMON_TOWER_5F_ITEM", c_uint8, 1),
        ("HS_POKEMON_TOWER_6F_ITEM_1", c_uint8, 1),
        ("HS_POKEMON_TOWER_6F_ITEM_2", c_uint8, 1),
        ("HS_POKEMON_TOWER_7F_ROCKET_1", c_uint8, 1),
        ("HS_POKEMON_TOWER_7F_ROCKET_2", c_uint8, 1),
        ("HS_POKEMON_TOWER_7F_ROCKET_3", c_uint8, 1),
        ("HS_POKEMON_TOWER_7F_MR_FUJI", c_uint8, 1),
        ("HS_MR_FUJIS_HOUSE_MR_FUJI", c_uint8, 1),
        ("HS_CELADON_MANSION_EEVEE_GIFT", c_uint8, 1),
        ("HS_GAME_CORNER_ROCKET", c_uint8, 1),
        ("HS_WARDENS_HOUSE_ITEM", c_uint8, 1),
        ("HS_POKEMON_MANSION_1F_ITEM_1", c_uint8, 1),
        ("HS_POKEMON_MANSION_1F_ITEM_2", c_uint8, 1),
        ("HS_FIGHTING_DOJO_GIFT_1", c_uint8, 1),
        ("HS_FIGHTING_DOJO_GIFT_2", c_uint8, 1),
        ("HS_SILPH_CO_1F_RECEPTIONIST", c_uint8, 1),
        ("HS_VOLTORB_1", c_uint8, 1),
        ("HS_VOLTORB_2", c_uint8, 1),
        ("HS_VOLTORB_3", c_uint8, 1),
        ("HS_ELECTRODE_1", c_uint8, 1),
        ("HS_VOLTORB_4", c_uint8, 1),
        ("HS_VOLTORB_5", c_uint8, 1),
        ("HS_ELECTRODE_2", c_uint8, 1),
        ("HS_VOLTORB_6", c_uint8, 1),
        ("HS_ZAPDOS", c_uint8, 1),
        ("HS_POWER_PLANT_ITEM_1", c_uint8, 1),
        ("HS_POWER_PLANT_ITEM_2", c_uint8, 1),
        ("HS_POWER_PLANT_ITEM_3", c_uint8, 1),
        ("HS_POWER_PLANT_ITEM_4", c_uint8, 1),
        ("HS_POWER_PLANT_ITEM_5", c_uint8, 1),
        ("HS_MOLTRES", c_uint8, 1),
        ("HS_VICTORY_ROAD_2F_ITEM_1", c_uint8, 1),
        ("HS_VICTORY_ROAD_2F_ITEM_2", c_uint8, 1),
        ("HS_VICTORY_ROAD_2F_ITEM_3", c_uint8, 1),
        ("HS_VICTORY_ROAD_2F_ITEM_4", c_uint8, 1),
        ("HS_VICTORY_ROAD_2F_BOULDER", c_uint8, 1),
        ("HS_BILL_POKEMON", c_uint8, 1),
        ("HS_BILL_1", c_uint8, 1),
        ("HS_BILL_2", c_uint8, 1),
        ("HS_VIRIDIAN_FOREST_ITEM_1", c_uint8, 1),
        ("HS_VIRIDIAN_FOREST_ITEM_2", c_uint8, 1),
        ("HS_VIRIDIAN_FOREST_ITEM_3", c_uint8, 1),
        ("HS_MT_MOON_1F_ITEM_1", c_uint8, 1),
        ("HS_MT_MOON_1F_ITEM_2", c_uint8, 1),
        ("HS_MT_MOON_1F_ITEM_3", c_uint8, 1),
        ("HS_MT_MOON_1F_ITEM_4", c_uint8, 1),
        ("HS_MT_MOON_1F_ITEM_5", c_uint8, 1),
        ("HS_MT_MOON_1F_ITEM_6", c_uint8, 1),
        ("HS_MT_MOON_B2F_FOSSIL_1", c_uint8, 1),
        ("HS_MT_MOON_B2F_FOSSIL_2", c_uint8, 1),
        ("HS_MT_MOON_B2F_ITEM_1", c_uint8, 1),
        ("HS_MT_MOON_B2F_ITEM_2", c_uint8, 1),
        ("HS_SS_ANNE_2F_RIVAL", c_uint8, 1),
        ("HS_SS_ANNE_1F_ROOMS_ITEM", c_uint8, 1),
        ("HS_SS_ANNE_2F_ROOMS_ITEM_1", c_uint8, 1),
        ("HS_SS_ANNE_2F_ROOMS_ITEM_2", c_uint8, 1),
        ("HS_SS_ANNE_B1F_ROOMS_ITEM_1", c_uint8, 1),
        ("HS_SS_ANNE_B1F_ROOMS_ITEM_2", c_uint8, 1),
        ("HS_SS_ANNE_B1F_ROOMS_ITEM_3", c_uint8, 1),
        ("HS_VICTORY_ROAD_3F_ITEM_1", c_uint8, 1),
        ("HS_VICTORY_ROAD_3F_ITEM_2", c_uint8, 1),
        ("HS_VICTORY_ROAD_3F_BOULDER", c_uint8, 1),
        ("HS_ROCKET_HIDEOUT_B1F_ITEM_1", c_uint8, 1),
        ("HS_ROCKET_HIDEOUT_B1F_ITEM_2", c_uint8, 1),
        ("HS_ROCKET_HIDEOUT_B2F_ITEM_1", c_uint8, 1),
        ("HS_ROCKET_HIDEOUT_B2F_ITEM_2", c_uint8, 1),
        ("HS_ROCKET_HIDEOUT_B2F_ITEM_3", c_uint8, 1),
        ("HS_ROCKET_HIDEOUT_B2F_ITEM_4", c_uint8, 1),
        ("HS_ROCKET_HIDEOUT_B3F_ITEM_1", c_uint8, 1),
        ("HS_ROCKET_HIDEOUT_B3F_ITEM_2", c_uint8, 1),
        ("HS_ROCKET_HIDEOUT_B4F_GIOVANNI", c_uint8, 1),
        ("HS_ROCKET_HIDEOUT_B4F_ITEM_1", c_uint8, 1),
        ("HS_ROCKET_HIDEOUT_B4F_ITEM_2", c_uint8, 1),
        ("HS_ROCKET_HIDEOUT_B4F_ITEM_3", c_uint8, 1),
        ("HS_ROCKET_HIDEOUT_B4F_ITEM_4", c_uint8, 1),
        ("HS_ROCKET_HIDEOUT_B4F_ITEM_5", c_uint8, 1),
        ("HS_SILPH_CO_2F_1", c_uint8, 1),
        ("HS_SILPH_CO_2F_2", c_uint8, 1),
        ("HS_SILPH_CO_2F_3", c_uint8, 1),
        ("HS_SILPH_CO_2F_4", c_uint8, 1),
        ("HS_SILPH_CO_2F_5", c_uint8, 1),
        ("HS_SILPH_CO_3F_1", c_uint8, 1),
        ("HS_SILPH_CO_3F_2", c_uint8, 1),
        ("HS_SILPH_CO_3F_ITEM", c_uint8, 1),
        ("HS_SILPH_CO_4F_1", c_uint8, 1),
        ("HS_SILPH_CO_4F_2", c_uint8, 1),
        ("HS_SILPH_CO_4F_3", c_uint8, 1),
        ("HS_SILPH_CO_4F_ITEM_1", c_uint8, 1),
        ("HS_SILPH_CO_4F_ITEM_2", c_uint8, 1),
        ("HS_SILPH_CO_4F_ITEM_3", c_uint8, 1),
        ("HS_SILPH_CO_5F_1", c_uint8, 1),
        ("HS_SILPH_CO_5F_2", c_uint8, 1),
        ("HS_SILPH_CO_5F_3", c_uint8, 1),
        ("HS_SILPH_CO_5F_4", c_uint8, 1),
        ("HS_SILPH_CO_5F_ITEM_1", c_uint8, 1),
        ("HS_SILPH_CO_5F_ITEM_2", c_uint8, 1),
        ("HS_SILPH_CO_5F_ITEM_3", c_uint8, 1),
        ("HS_SILPH_CO_6F_1", c_uint8, 1),
        ("HS_SILPH_CO_6F_2", c_uint8, 1),
        ("HS_SILPH_CO_6F_3", c_uint8, 1),
        ("HS_SILPH_CO_6F_ITEM_1", c_uint8, 1),
        ("HS_SILPH_CO_6F_ITEM_2", c_uint8, 1),
        ("HS_SILPH_CO_7F_1", c_uint8, 1),
        ("HS_SILPH_CO_7F_2", c_uint8, 1),
        ("HS_SILPH_CO_7F_3", c_uint8, 1),
        ("HS_SILPH_CO_7F_4", c_uint8, 1),
        ("HS_SILPH_CO_7F_RIVAL", c_uint8, 1),
        ("HS_SILPH_CO_7F_ITEM_1", c_uint8, 1),
        ("HS_SILPH_CO_7F_ITEM_2", c_uint8, 1),
        ("HS_SILPH_CO_7F_8", c_uint8, 1),
        ("HS_SILPH_CO_8F_1", c_uint8, 1),
        ("HS_SILPH_CO_8F_2", c_uint8, 1),
        ("HS_SILPH_CO_8F_3", c_uint8, 1),
        ("HS_SILPH_CO_9F_1", c_uint8, 1),
        ("HS_SILPH_CO_9F_2", c_uint8, 1),
        ("HS_SILPH_CO_9F_3", c_uint8, 1),
        ("HS_SILPH_CO_10F_1", c_uint8, 1),
        ("HS_SILPH_CO_10F_2", c_uint8, 1),
        ("HS_SILPH_CO_10F_3", c_uint8, 1),
        ("HS_SILPH_CO_10F_ITEM_1", c_uint8, 1),
        ("HS_SILPH_CO_10F_ITEM_2", c_uint8, 1),
        ("HS_SILPH_CO_10F_ITEM_3", c_uint8, 1),
        ("HS_SILPH_CO_11F_1", c_uint8, 1),
        ("HS_SILPH_CO_11F_2", c_uint8, 1),
        ("HS_SILPH_CO_11F_3", c_uint8, 1),
        ("HS_UNUSED_MAP_F4_1", c_uint8, 1),
        ("HS_POKEMON_MANSION_2F_ITEM", c_uint8, 1),
        ("HS_POKEMON_MANSION_3F_ITEM_1", c_uint8, 1),
        ("HS_POKEMON_MANSION_3F_ITEM_2", c_uint8, 1),
        ("HS_POKEMON_MANSION_B1F_ITEM_1", c_uint8, 1),
        ("HS_POKEMON_MANSION_B1F_ITEM_2", c_uint8, 1),
        ("HS_POKEMON_MANSION_B1F_ITEM_3", c_uint8, 1),
        ("HS_POKEMON_MANSION_B1F_ITEM_4", c_uint8, 1),
        ("HS_POKEMON_MANSION_B1F_ITEM_5", c_uint8, 1),
        ("HS_SAFARI_ZONE_EAST_ITEM_1", c_uint8, 1),
        ("HS_SAFARI_ZONE_EAST_ITEM_2", c_uint8, 1),
        ("HS_SAFARI_ZONE_EAST_ITEM_3", c_uint8, 1),
        ("HS_SAFARI_ZONE_EAST_ITEM_4", c_uint8, 1),
        ("HS_SAFARI_ZONE_NORTH_ITEM_1", c_uint8, 1),
        ("HS_SAFARI_ZONE_NORTH_ITEM_2", c_uint8, 1),
        ("HS_SAFARI_ZONE_WEST_ITEM_1", c_uint8, 1),
        ("HS_SAFARI_ZONE_WEST_ITEM_2", c_uint8, 1),
        ("HS_SAFARI_ZONE_WEST_ITEM_3", c_uint8, 1),
        ("HS_SAFARI_ZONE_WEST_ITEM_4", c_uint8, 1),
        ("HS_SAFARI_ZONE_CENTER_ITEM", c_uint8, 1),
        ("HS_CERULEAN_CAVE_2F_ITEM_1", c_uint8, 1),
        ("HS_CERULEAN_CAVE_2F_ITEM_2", c_uint8, 1),
        ("HS_CERULEAN_CAVE_2F_ITEM_3", c_uint8, 1),
        ("HS_MEWTWO", c_uint8, 1),
        ("HS_CERULEAN_CAVE_B1F_ITEM_1", c_uint8, 1),
        ("HS_CERULEAN_CAVE_B1F_ITEM_2", c_uint8, 1),
        ("HS_VICTORY_ROAD_1F_ITEM_1", c_uint8, 1),
        ("HS_VICTORY_ROAD_1F_ITEM_2", c_uint8, 1),
        ("HS_CHAMPIONS_ROOM_OAK", c_uint8, 1),
        ("HS_SEAFOAM_ISLANDS_1F_BOULDER_1", c_uint8, 1),
        ("HS_SEAFOAM_ISLANDS_1F_BOULDER_2", c_uint8, 1),
        ("HS_SEAFOAM_ISLANDS_B1F_BOULDER_1", c_uint8, 1),
        ("HS_SEAFOAM_ISLANDS_B1F_BOULDER_2", c_uint8, 1),
        ("HS_SEAFOAM_ISLANDS_B2F_BOULDER_1", c_uint8, 1),
        ("HS_SEAFOAM_ISLANDS_B2F_BOULDER_2", c_uint8, 1),
        ("HS_SEAFOAM_ISLANDS_B3F_BOULDER_1", c_uint8, 1),
        ("HS_SEAFOAM_ISLANDS_B3F_BOULDER_2", c_uint8, 1),
        ("HS_SEAFOAM_ISLANDS_B3F_BOULDER_3", c_uint8, 1),
        ("HS_SEAFOAM_ISLANDS_B3F_BOULDER_4", c_uint8, 1),
        ("HS_SEAFOAM_ISLANDS_B4F_BOULDER_1", c_uint8, 1),
        ("HS_SEAFOAM_ISLANDS_B4F_BOULDER_2", c_uint8, 1),
        ("HS_ARTICUNO", c_uint8, 1),
    ]


class MissableFlags(Union):
    # These missable flags is a 32 byte object
    _fields_ = [("b", MissableFlagsBits), ("asbytes", c_uint8 * 32)]

    def __init__(self):
        super().__init__()
        emu = get_pyboy_instance()
        self.asbytes = (c_uint8 * 32)(*emu.memory[0xD5A6 : 0xD5A6 + 32])

    def get_missable(self, missable: str) -> bool:
        return bool(getattr(self.b, missable))