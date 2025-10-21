# lightcraft/__main__.py

"""
LightCraft package entrypoint.

This script enables LightCraft to be run directly from the command line
using the command:  lightcraft
"""

from .main_view import run

def main():
    """Launch the LightCraft runtime."""
    run()

if __name__ == "__main__":
    main()
