pokegym is to be used in conjunction with pufferlib


https://github.com/xinpw8/pokegym.git -b pokegym_bill

https://github.com/xinpw8/PufferLib.git -b puffer_BET_bill

In a nutshell, right now, we have the AI starting from a state post-Oak-parcel-fetch-quest. 
It goes to Cerulean City and beats Misty, but gets stuck on Bill. Afaik, it's not talking to PCs, nor to Bill. 
You can clone my repositories at

https://github.com/xinpw8/pokegym.git -b pokegym_bill

https://github.com/xinpw8/PufferLib.git -b puffer_BET_bill

Recommended installation:
Clone both into a main folder, [mv PufferLib pufferlib] (because AnNoYiNg CaSe).
Then we'll install both locally (i.e. in developer mode):

cd pufferlib
pip install -e ".[cleanrl,pokemon_red]"
cd ..
pip install -e pokegym

Then you need to put the pokemonred.gb rom file into the pufferlib folder and that's it. 
Environment file to edit is in pokegym/pokegym. 

To run, cd pufferlib, in demo.py, ctrl+f xinpw8 and change to your wandb account name, then:

python demo.py --train --track --vectorization multiprocessing

Default max global steps is in config.py, set to 100m. 
Current features are save/load states, LSTM, explore (buffered view of area around agent), and NPC rewards.
Gets past SS Anne. Unclear as to whether it gets or teaches any pokemon HM01 Cut.

https://wandb.ai/xinpw8/pufferlib/runs/bill_new_npc_script_2?workspace=user-xinpw8

![image](https://github.com/xinpw8/pokegym/assets/38776436/29557131-d7ea-4bb3-b215-c499c5ec86e7)
