Commands for new project structure
1. ``python -m data.raw_data.punch_label_gen.analyze.stats``
 
2. ``python -m data.neural_data_prep.extract -n test -tr 5 6 7 8 -tw 9``
``python -m data.neural_data_prep.extract -n test2``
Flags: -d for dev mode and -l for local machine development

3. ``python -m train.start -n test -tr 5 6 7 8 -tw 9``  
``python -m train.start -d -l -n test2 ``  
Flags: -d for dev mode and -l for local machine development

4. ``python -m vis.backend.server.flask.launch``