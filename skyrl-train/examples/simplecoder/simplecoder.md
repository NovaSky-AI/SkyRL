# SimpleCoder

This is a simple coding environment that allows solving SWE bench like coding challenges.

It uses a basic sandbox environment powered by Bubblewrap (https://github.com/containers/bubblewrap).

## How to use it

The docker image / environment you are working in needs to have guix installed, you can
e.g. install it by running

```shell
sudo apt install bubblewrap
```

To run the example, first clone the test repository
```shell
git clone https://github.com/SWE-agent/test-repo
```

and then run
```shell
python simplecoder.py
```

*Disclaimer*: The integration with SkyRL Train is still ongoing.

