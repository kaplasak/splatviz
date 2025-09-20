import click
from splatviz import Splatviz


@click.command()
@click.option("--mode", help="[default, attach]", default="default")
@click.option("--host", help="host address", default="127.0.0.1")
@click.option("--port", help="port", default=6007)
def main(mode, host, port):
    splatviz = Splatviz(mode=mode, host=host, port=port)
    while not splatviz.should_close():
        splatviz.draw_frame()
    splatviz.close()


if __name__ == "__main__":
    main()
