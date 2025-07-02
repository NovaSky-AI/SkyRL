Developing SkyRL-Gym across SkyRL-Train
=======================================
To develop both SkyRL-Gym and SkyRL-Train in parallel, you must symlink the gym package into the train package for 
your most up to date code to be shipped with the ray working directory.

First symlink the gym package into the train package:

.. code-block:: bash
    cd skyrl-train
    ln -s ../skyrl-gym/skyrl_gym/ skyrl-gym

Then the following to your ``pyproject.toml`` file:

.. code-block:: toml

    [tool.uv.sources]
    skyrl-gym = { path = "./skyrl-gym", editable = true }

Now make sure that your uv.lock file is up to date, and isn't resolving to the wrong version of the gym package.

.. code-block:: bash

    uv sync

Now you're all set! Changes made in SkyRL-Gym will be shipped with the ray working directory when you run the trainer in SkyRL-Train. 