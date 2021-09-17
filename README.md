Commands for new project structure
1. ``python -m data.raw_data.punch_label_gen.analyze.stats``
 
2. ``python -m data.neural_data_prep.extract_multiple -n root_tr_exp -tr 5 6 7 8 9 10 -tw 5 ``
``python -m data.neural_data_prep.extract_multiple -n wrist_tr_exp -tr 5 -tw 5 6 7 8 9 10``
``python -m data.neural_data_prep.extract_multiple -n test -tr 5 6 7 8 -tw 9``
``python -m data.neural_data_prep.extract_multiple -n test2``
``python -m data.neural_data_prep.extract -d -l``
Flags: -d for dev mode and -l for local machine development

3. ``python -m train.start_multiple -n root_tr_exp -tr 5 6 7 8 9 10 -tw 5``
``python -m train.start_multiple -n wrist_tr_exp -tr 5 -tw 5 6 7 8 9 10``
``python -m train.start_multiple -n test -tr 5 6 7 8 -tw 9``  
``python -m train.start_multiple -d -l -n test2 ``  
``python -m train.start -d -l``  
Flags: -d for dev mode and -l for local machine development

4. ``python -m vis.backend.server.flask.launch``
5. ``python -m miscellaneous.download_exp_folder -e wrist_tr_exp``
6. ``python -m miscellaneous.download -m  fr_1_tr_5_5_ep_300/2021-09-14_13-15-08``