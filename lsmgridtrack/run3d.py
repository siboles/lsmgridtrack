import argparse
from typing import Optional
import pathlib

from lsmgridtrack import config, image, registration, kinematics


def main(
    config_path: str,
    reference_path: str,
    deformed_path: str,
    vtk_out: str,
    excel_out: Optional[str],
    ref2vtk: Optional[str],
    def2vtk: Optional[str],
):
    options = config.parse_config(config_path)
    if pathlib.Path(reference_path).is_dir():
        reference_image = image.parse_image_sequence(reference_path, options.image)
    else:
        reference_image = image.parse_image_file(reference_path, options.image)
    if ref2vtk:
        image.write_image_as_vtk(reference_image, ref2vtk)

    if pathlib.Path(deformed_path).is_dir():
        deformed_image = image.parse_image_sequence(deformed_path, options.image)
    else:
        deformed_image = image.parse_image_file(deformed_path, options.image)
    if def2vtk:
        image.write_image_as_vtk(deformed_image, def2vtk)

    reg = registration.create_registration(options.registration, reference_image)
    transform = registration.register(reg, reference_image, deformed_image)
    if options.registration.final_landmark_transform:
        transform = registration.apply_final_landmark_transform(
            reference_image, deformed_image, transform, options.registration
        )
    results = kinematics.get_kinematics(options.grid, options.image, transform)
    kinematics.write_kinematics_to_vtk(results, vtk_out)
    if excel_out:
        kinematics.write_kinematics_to_excel(results, excel_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, nargs=1, help="Path to configuration file.")
    parser.add_argument(
        "--reference",
        type=str,
        nargs=1,
        help="Path to reference image file or image file sequence.",
    )
    parser.add_argument(
        "--deformed",
        type=str,
        nargs=1,
        help="Path to deformed image file or image file sequence.",
    )
    parser.add_argument(
        "--vtk",
        type=str,
        nargs="?",
        default="output",
        help="Base name of file to write vtk grid.",
    )
    parser.add_argument(
        "--excel",
        type=str,
        nargs="?",
        default=None,
        help="Base name excel file to write.",
    )
    parser.add_argument(
        "--ref2vtk",
        type=str,
        nargs="?",
        default=None,
        help="Write reference image to vtk file with provided name.",
    )
    parser.add_argument(
        "--def2vtk",
        type=str,
        nargs="?",
        default=None,
        help="Write deformed image to vtk file with provided name.",
    )

    args = parser.parse_args()

    main(
        config_path=args.config[0],
        reference_path=args.reference[0],
        deformed_path=args.deformed[0],
        vtk_out=args.vtk,
        excel_out=args.excel,
        ref2vtk=args.ref2vtk,
        def2vtk=args.def2vtk,
    )
