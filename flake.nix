{
  description = "Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            pkgs.python313
            pkgs.python313Packages.jupyterlab
            pkgs.python313Packages.pytorch
            pkgs.python313Packages.numpy
            pkgs.python313Packages.tqdm
            pkgs.python313Packages.matplotlib
          ];

          shellHook = ''
            devenv working
          '';
        };
      });
}