{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [
    pkgs.poppler_utils  # pdftoppm, used by pdf2image
    pkgs.resvg
    (pkgs.texlive.combine {
      inherit (pkgs.texlive)
        scheme-basic
        standalone
        pgf
        pgfplots
        amsmath
        amscls
        collection-pictures
        collection-latexrecommended;
    })
  ];
}
