"""CLI wrapper for QProfiler that injects the default config directory."""
import sys
import os

def main():
    """Run QProfiler with the bundled default config directory if none is specified."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_dir = os.path.join(script_dir, 'configs')

    has_config_dir = any('--config-dir' in arg for arg in sys.argv)
    has_config_path = any('--config-path' in arg for arg in sys.argv)

    if not has_config_dir and not has_config_path:
        sys.argv.insert(1, f'--config-dir={os.path.abspath(default_config_dir)}')

    from qbiocode.apps.qprofiler.qprofiler import main as qprofiler_main
    qprofiler_main()

if __name__ == '__main__':
    main()
