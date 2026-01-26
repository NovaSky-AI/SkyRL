import logging
import os
from typing import Dict

logger = logging.getLogger("desktopenv.getters.gimp")


def get_gimp_config_file(env, config: Dict[str, str]):
    """
    Gets the config setting of GIMP.
    """

    os_type = env.vm_platform
    print(os_type)

    if os_type == "Linux":
        try:
            # First check if the GIMP config directory exists
            check_command = f"import os; print(os.path.exists(os.path.expanduser('~/.config/GIMP/2.10/{config['file_name']}')))"
            exists_result = env.controller.execute_python_command(check_command)
            
            if exists_result.get('output', '').strip().lower() == 'false':
                logger.warning(f"GIMP config file {config['file_name']} does not exist. Skipping GIMP config retrieval.")
                # Create an empty placeholder file to prevent evaluation errors
                _path = os.path.join(env.cache_dir, config["dest"])
                os.makedirs(os.path.dirname(_path), exist_ok=True)
                with open(_path, "w") as f:
                    f.write("# GIMP config file not found - placeholder\n")
                return _path
            
            config_path = \
                env.controller.execute_python_command(f"import os; print("
                                                      f"os"
                                                      f".path.expanduser("
                                                      f"'~/.config/GIMP/2.10/"
                                                      f"{config['file_name']}'))")[
                    'output'].strip()
        except Exception as e:
            logger.error(f"Failed to check GIMP config file existence: {e}")
            # Create an empty placeholder file to prevent evaluation errors
            _path = os.path.join(env.cache_dir, config["dest"])
            os.makedirs(os.path.dirname(_path), exist_ok=True)
            with open(_path, "w") as f:
                f.write("# GIMP config file check failed - placeholder\n")
            return _path
    # TODO: Add support for macOS and Windows
    else:
        raise Exception("Unsupported operating system", os_type)

    _path = os.path.join(env.cache_dir, config["dest"])
    
    try:
        content = env.controller.get_file(config_path)
        
        if not content:
            logger.warning("Failed to get GIMP config file content. Creating placeholder.")
            # Create an empty placeholder file to prevent evaluation errors
            os.makedirs(os.path.dirname(_path), exist_ok=True)
            with open(_path, "w") as f:
                f.write("# GIMP config file content not available - placeholder\n")
            return _path
        
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        with open(_path, "wb") as f:
            f.write(content)
        
        return _path
        
    except Exception as e:
        logger.error(f"Error retrieving GIMP config file: {e}")
        # Create an empty placeholder file to prevent evaluation errors
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        with open(_path, "w") as f:
            f.write(f"# GIMP config file retrieval failed: {e} - placeholder\n")
        return _path
