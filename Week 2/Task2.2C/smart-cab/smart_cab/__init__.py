from gym.envs.registration import register

register(
    id='smart_cab-v1',
    entry_point='smart_cab.envs:TaxiEnv')
