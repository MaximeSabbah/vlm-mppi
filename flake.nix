{
  description = "VLM-MPPI: VLM planner for safe human-robot interaction";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;   # needed for CUDA
          config.cudaSupport = true;
        };

        pythonPkgs = pkgs.python310Packages;

      in {
        devShells.default = pkgs.mkShell {
          name = "vlm-mppi";

          buildInputs = with pkgs; [
            python310
            python310Packages.pip
            python310Packages.virtualenv

            # System deps for opencv, torch, etc.
            stdenv.cc.cc.lib
            zlib
            libGL
            glib

            # Useful dev tools
            git
            curl
            jq
          ];

          shellHook = ''
            echo "🤖 VLM-MPPI development environment"
            echo ""

            # Create venv if it doesn't exist
            if [ ! -d .venv ]; then
              echo "Creating Python virtual environment..."
              python -m venv .venv
            fi
            source .venv/bin/activate

            # Install deps if needed
            if ! python -c "import transformers" 2>/dev/null; then
              echo "Installing Python dependencies..."
              pip install -q -r requirements.txt
            fi

            echo "Python: $(python --version)"
            echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'not installed yet')"
            echo ""
            echo "Run examples:"
            echo "  python examples/01_test_vlm_basic.py --image scene.jpg"
            echo "  python examples/02_vlm_keypoint_planner.py --image scene.jpg"
          '';

          # For CUDA / torch to find libs
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.libGL
            pkgs.glib
          ];
        };
      });
}
