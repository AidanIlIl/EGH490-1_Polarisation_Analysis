import os
import cv2

from analysis import PolarizationProcessor


def process_image(
    input_path,
    output_dir,
    filter_name='None',
    kernel=5,
    sigma=1.0,
    pol_type='linear',
    roi=[200, 200, 6000, 6000],
    file_params=None,
    oversat=0.0,
    undersat=0.0,
    save_demosaic=True,
    save_graphs=True,
    save_csv=True,
    prefix='result',
):
    os.makedirs(output_dir, exist_ok=True)
    processor = PolarizationProcessor()
    if not processor.load_image(input_path):
        raise FileNotFoundError(f"Unable to load image: {input_path}")

    processor.apply_filter(filter_name, kernel=kernel, sigma=sigma)
    processor.calculate(pol_type=pol_type, roi=roi)

    saved_paths = {}
    image_path = os.path.join(output_dir, f'{prefix}.png')
    processor.save_image(image_path)
    saved_paths['image'] = image_path

    if save_demosaic:
        demosaic_path = os.path.join(output_dir, f'{prefix}_demosaic.png')
        processor.save_demosaic(demosaic_path)
        saved_paths['demosaic'] = demosaic_path

    if save_graphs:
        graph_dir = os.path.join(output_dir, f'{prefix}_graphs')
        processor.save_polarization_outputs(
            graph_dir,
            prefix=prefix,
            file_params=file_params,
            oversat=oversat,
            undersat=undersat,
            save_csv=save_csv,
        )
        saved_paths['graphs'] = graph_dir
        if save_csv:
            saved_paths['csv'] = os.path.join(graph_dir, f'{prefix}.csv')
    elif save_csv:
        csv_path = os.path.splitext(os.path.join(output_dir, f'{prefix}.csv'))[0]
        processor.export_csv(csv_path, file_params=file_params, oversat=oversat, undersat=undersat)
        saved_paths['csv'] = f'{csv_path}.csv'

    return saved_paths


def batch_vary_gaussian_blur(input_path, output_dir, kernels, sigmas, pol_type='linear'):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for kernel in kernels:
        for sigma in sigmas:
            prefix = f'blur_k{kernel}_s{sigma:.2f}'
            item_dir = os.path.join(output_dir, prefix)
            saved = process_image(
                input_path,
                item_dir,
                filter_name='Gaussian Blur',
                kernel=kernel,
                sigma=sigma,
                pol_type=pol_type,
                prefix=prefix,
            )
            results.append(saved)
    return results


def main():
    input_path = './Tests/Image20260421203409.png'
    output_dir = './batch_outputs'
    kernels = [3, 5, 7, 9]
    sigmas = [0.5, 1.0, 2.0]
    batch_vary_gaussian_blur(input_path, output_dir, kernels, sigmas)
    print(f"Saved blurred images to {output_dir}")


if __name__ == '__main__':
    main()
