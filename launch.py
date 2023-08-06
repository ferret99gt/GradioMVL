from modules import launch_utils


python = launch_utils.python
index_url = launch_utils.index_url

run = launch_utils.run
is_installed = launch_utils.is_installed

run_pip = launch_utils.run_pip
check_run_python = launch_utils.check_run_python
prepare_environment = launch_utils.prepare_environment
start = launch_utils.start

def main():
    prepare_environment()

    start()


if __name__ == "__main__":
    main()
