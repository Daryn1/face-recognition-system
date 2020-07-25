/*
	This is the main part of the code of our face recognition system. It includes 
	the creation of the graphical user interface and the implementation of the face 
	recognition algorithm. 
	
	The algorithm will be described in our paper. In short, the algorithm is based on 
	the estimation of the face rotation angle. Using this angle, a face image with the 
	same rotation angle is retrieved from the face database. Comparing face images with 
	close rotation angles can reduce the effect of face pose on recognition accuracy.
*/

// pch.h is required for successful compilation. We do not use pre-compiled headers
#include "pch.h"

// Include standard libraries 
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <math.h>

// Include header files of dlib
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/threads.h>
#include <dlib/string.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

// Include header files of OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

// Include header files of MXNet
#include <mxnet-cpp/MxNetCpp.h>
#include <mxnet/c_predict_api.h>

// Include header files of MTCNN face detector and MobileFaceNet CNN
#include "face_align.hpp"
#include "mxnet_mtcnn.hpp"
#include "feature_extract.hpp"
#include "comm_lib.hpp"

/*
	This object represents the main window of graphical user interface of the face recognition system. 
	It inherits dlib::drawable_window and dlib::threaded_object so it is is capable of containing 
	drawable objects and multithreading.
*/
class main_window : public dlib::drawable_window, public dlib::threaded_object
{
public:

	// This object represents the window that is used to add faces to the face database
	class add_window : public dlib::drawable_window
	{
	public:

		add_window(
		) : // All widgets take their parent window as an argument to their constructor
			label1(*this),
			label2(*this),
			label3(*this),
			add_button(*this),
			input_image_widget(*this),
			text_field1(*this),
			list_box1(*this)
		{
			// Set the size, position and title of this window
			set_size(600, 650);
			set_pos(10, 10);
			set_title("Add face");

			// Set the positions, sizes, texts of the widgets
			input_image_widget.set_pos(225, 100);
			text_field1.set_pos(50, 325);
			text_field1.set_width(300);
			text_field1.set_text("Daryn");
			list_box1.set_pos(50, text_field1.top() + 103);
			list_box1.set_size(300, 175);
			label1.set_pos(input_image_widget.left() - 5, 50 - 13);
			label1.set_text("Input face:");
			label2.set_pos(27, text_field1.top() - 30 - 11);
			label2.set_text("Person's name:");
			label3.set_pos(27, list_box1.top() - 30 - 11);
			label3.set_text("Database:");
			add_button.set_pos(text_field1.left() + text_field1.width() + 50, text_field1.top());
			add_button.set_name("Add  ");
			add_button.set_size(150, text_field1.height());

			// Set a function to be called when the user clicks on the widget
			list_box1.set_click_handler(*this, &add_window::change_text_field);
		}

		// Object destructor 
		~add_window()
		{
			// We should always call close_window() in the destructor of window objects to ensure
			// that no events will be sent to this window while it is being destructed 
			close_window();
		}

	private:

		// We should be able to access private members of this window from the main window
		friend main_window;

		// This function changes the text of text_field1 to the name of a person from the face database 
		// when the user selects the name from list_box1
		void change_text_field(unsigned long index)
		{
			text_field1.set_text(person_names[index]);
		}

		// Declare all add window widgets
		dlib::label label1;
		dlib::label label2;
		dlib::label label3;
		dlib::image_widget input_image_widget;
		dlib::text_field text_field1;
		dlib::list_box list_box1;
		dlib::button add_button;

		// Declare an array that stores the names of people in the face database
		dlib::array<std::string> person_names;
	
	};

	// Object constructor
	main_window() : // All widgets take their parent window as an argument to their constructor.
		input_image_label(*this),
		detected_face_label(*this),
		predicted_face_label(*this),
		predicted_person_name_label_1(*this),
		predicted_face_angle(*this),
		find_faces_button(*this),
		recognize_button(*this),
		snapshot_button(*this),
		capture_video_button(*this),
		add_to_database_button(*this),
		menu_bar1(*this),
		input_image_widget(*this),
		detected_face_image_widget(*this),
		predicted_face_image_widget(*this)
	{
		// Set the size and name of the main window
		set_size(1600, 900);
		set_title("Face detector");

		// Load the fonts used in the GUI
		std::ifstream font_file1("./arial_36.bdf");
		std::ifstream font_file2("./arial_32.bdf");
		arial_36.read_bdf_file(font_file1, 255);
		parial_36.reset(&arial_36);
		arial_32.read_bdf_file(font_file2, 255);
		parial_32.reset(&arial_32);

		// Set loaded fonts to the widgets
		add_window1.text_field1.set_main_font(parial_32);
		add_window1.list_box1.set_main_font(parial_32);
		add_window1.label1.set_main_font(parial_36);
		add_window1.label2.set_main_font(parial_32);
		add_window1.label3.set_main_font(parial_32);
		add_window1.add_button.set_main_font(parial_32);

		// Set the positions of the image widgets
		input_image_widget.set_pos(50, 100);
		detected_face_image_widget.set_pos(835, 100);
		predicted_face_image_widget.set_pos(1200, 100);

		// Define the size of all buttons
		unsigned long buttons_width = 250;
		unsigned long buttons_height = 80;
		unsigned long dist_b_b = 50;

		// Set up all buttons
		snapshot_button.set_main_font(parial_32);
		snapshot_button.set_pos(50, 750);
		snapshot_button.set_name("Snapshot  ");
		snapshot_button.set_size(buttons_width, buttons_height);
		capture_video_button.set_main_font(parial_32);
		capture_video_button.set_pos(snapshot_button.left(), snapshot_button.top());
		capture_video_button.set_name("Capture video  ");
		capture_video_button.set_size(buttons_width, buttons_height);
		capture_video_button.hide();
		find_faces_button.set_main_font(parial_32);
		find_faces_button.set_pos(snapshot_button.left() + buttons_width + dist_b_b, 750);
		find_faces_button.set_name("Find faces  ");
		find_faces_button.set_size(buttons_width, buttons_height);
		find_faces_button.disable();
		recognize_button.set_main_font(parial_32);
		recognize_button.set_pos(find_faces_button.left() + buttons_width + dist_b_b, 750);
		recognize_button.set_name("Recognize  ");
		recognize_button.set_size(buttons_width, buttons_height);
		recognize_button.disable();
		add_to_database_button.set_main_font(parial_32);
		add_to_database_button.set_pos(recognize_button.left() + buttons_width + dist_b_b, 750);
		add_to_database_button.set_name("Add to database ");
		add_to_database_button.set_size(buttons_width + 50, buttons_height);
		add_to_database_button.disable();
		add_window1.add_button.set_size(150, add_window1.text_field1.height());

		// Set up all labels
		input_image_label.set_main_font(parial_36);
		input_image_label.set_pos(input_image_widget.left() + 320 - 95, input_image_widget.top() - 50);
		input_image_label.set_text("Input image:");
		detected_face_label.set_main_font(parial_36);
		detected_face_label.set_pos(detected_face_image_widget.left() - 39, detected_face_image_widget.top() - 50);
		detected_face_label.set_text("Detected face:");
		predicted_face_label.set_main_font(parial_36);
		predicted_face_label.set_pos(predicted_face_image_widget.left() - 62, predicted_face_image_widget.top() - 50);
		predicted_face_label.set_text("Predicted person:");
		predicted_person_name_label_1.set_main_font(parial_36);
		predicted_person_name_label_1.set_pos(predicted_face_label.left(), predicted_face_label.top() + 225);
		predicted_person_name_label_1.set_text(" ");
		predicted_face_angle.set_main_font(parial_36);
		predicted_face_angle.set_pos(detected_face_label.left(), detected_face_label.top() + 225);
		predicted_face_angle.set_text(" ");

		// Set functions to be called when the user clicks on the buttons
		snapshot_button.set_click_handler(*this, &main_window::on_snapshot_button_clicked);
		capture_video_button.set_click_handler(*this, &main_window::on_capture_video_button_clicked);
		find_faces_button.set_click_handler(*this, &main_window::on_find_faces_button_clicked);
		recognize_button.set_click_handler(*this, &main_window::on_recognize_button_clicked);
		add_to_database_button.set_click_handler(*this, &main_window::show_add_window);
		add_window1.add_button.set_click_handler(*this, &main_window::add_to_face_database);

		//--- MAKE A MENU BAR
		// First we set the number of menus we want in our menu bar
		menu_bar1.set_number_of_menus(3);

		// Now we set the name of our menu. The 'F' means that the F in Menu will be underlined
		// and the user will be able to select it by hitting alt+F
		menu_bar1.set_menu_name(0, "File", 'F');
		menu_bar1.set_menu_name(1, "Tools", 's');
		menu_bar1.set_menu_name(2, "Help", 'H');

		// Add items to the menu. Items in a menu are listed in the order in which they were added
		// Make a menu item that calls open_file function when the user selects it.
		menu_bar1.menu(0).add_menu_item(dlib::menu_item_text("Open", *this, &main_window::open_file, 'O'));

		// Add a horizontal separating line to the menu
		menu_bar1.menu(0).add_menu_item(dlib::menu_item_separator());

		// Do the same for the rest of the items
		menu_bar1.menu(0).add_menu_item(dlib::menu_item_text("Save As...", *this, &main_window::open_save_file_box, 'S'));
		menu_bar1.menu(1).add_menu_item(dlib::menu_item_text("Snapshot", *this, &main_window::on_snapshot_button_clicked, 'S'));
		menu_bar1.menu(1).add_menu_item(dlib::menu_item_separator());
		menu_bar1.menu(1).add_menu_item(dlib::menu_item_text("Find faces", *this, &main_window::on_find_faces_button_clicked, 'F'));
		menu_bar1.menu(1).add_menu_item(dlib::menu_item_separator());
		menu_bar1.menu(1).add_menu_item(dlib::menu_item_text("Recognize", *this, &main_window::on_recognize_button_clicked, 'R'));
		menu_bar1.menu(1).add_menu_item(dlib::menu_item_separator());
		menu_bar1.menu(1).add_menu_item(dlib::menu_item_text("Add to database", *this, &main_window::show_add_window, 'A'));
		menu_bar1.menu(2).add_menu_item(dlib::menu_item_text("About", *this, &main_window::show_about, 'A'));

		// Show the main window
		show();

		// Load the face detector model
		std::string mtcnn_model = "./models/mtcnn";
		face_detector.LoadModule(mtcnn_model);

		// Load the CNN model responsible for face recognition.
		feature_extractor.LoadExtractModule("./models/mobilefacenet/model-0000.params", "./models/mobilefacenet/model-symbol.json", 1, 3, 112, 112);

		//--- LOAD THE FACE DATABASE
		dlib::directory face_database_dir("./face_database/");
		std::vector<dlib::directory> dirs = face_database_dir.get_dirs();
		for (unsigned long i = 0; i < dirs.size(); i++)
		{
			// Add an empty entry to the face database  
			face_database.push_back(Face_data());

			// Copy the directory name to the person name in the database
			std::string temp = dirs[i].name();
			face_database[i].person_name = temp;
			add_window1.person_names.push_back(temp);

			// Open .xml file containing feature vectors of the person
			cv::FileStorage feature_vectors_storage(face_database_dir.full_name() + "/" + dirs[i].name() + "/feature_vectors.xml", cv::FileStorage::READ);
			if (!feature_vectors_storage.isOpened())
			{
				std::cerr << "failed to open filestorage" << std::endl;
			}

			// Load feature vectors into the face database
			feature_vectors_storage["feature_vector_90"] >> face_database[i].feature_vector_90;
			feature_vectors_storage["feature_vector_45"] >> face_database[i].feature_vector_45;
			feature_vectors_storage["feature_vector_0"] >> face_database[i].feature_vector_0;
			feature_vectors_storage["feature_vector_m45"] >> face_database[i].feature_vector_m45;
			feature_vectors_storage["feature_vector_m90"] >> face_database[i].feature_vector_m90;

			// Load the face image of the person
			face_database[i].face_image = cv::imread(face_database_dir.full_name() + "/" + dirs[i].name() + "/0001.png");
		}

		// Open the default camera using default API
		cap.open(0);

		// Check if we have successfully connected to the camera
		if (cap.isOpened())
		{
			// Set the preferable frame size
			int preferable_camera_width = 640;
			int preferable_camera_height = 480;
			cap.set(cv::CAP_PROP_FRAME_WIDTH, preferable_camera_width);
			cap.set(cv::CAP_PROP_FRAME_HEIGHT, preferable_camera_height);

			// Start our thread going in the thread() function
			start();
		}
		else { // If there is no connection to the camera
			std::cerr << "Unable to connect to the camera\n";

			// Stop our thread
			stop();

			// Disable the button snapshot
			snapshot_button.disable();
		}
	}

	

	// Object destructor
	~main_window()
	{
		// We should always call close_window() in the destructor of window
		// objects to ensure that no events will be sent to this window while 
		// it is being destructed
		close_window();

		// Tell the thread() function to stop.  This will cause should_stop() to 
		// return true so the thread knows what to do
		stop();

		// Wait for the thread to stop before letting this object destruct itself.
		// Also we are required to wait for the thread to end before 
		// letting this object destruct itself
		wait();
	}

private:

	// We should be able to access private members of the main window from the add window
	friend add_window;

	// This function stops grabbing frames from the camera and saves the last grabbed frame
	void on_snapshot_button_clicked()
	{
		/*
			Stop the thread so that the program stops updating the input image
			Now it works like this: somewhere in thread() -> button clicked() -> continue thread()
		*/
		stop();

		// Hide the button snapshot
		snapshot_button.hide();

		// Grab a frame from the camera 5 times without delay
		cv::Mat input_image;
		for (int i = 0; i < 5; i++)
		{
			cap.read(input_image);
		}

		//--- SAVE THE INPUT IMAGE ON DISK
		dlib::create_directory("snapshots");
		dlib::directory snapshots_dir("./snapshots/");
		std::vector<dlib::file> files = snapshots_dir.get_files();
		std::string filename;
		int number_of_images(1);
		if (files.size() > 0)
		{ // If the folder is not empty
			sort(files.begin(), files.end());
			filename = files[files.size() - 1].name();
			filename.erase(filename.length() - 4);
			number_of_images = stoi(filename) + 1;
			if (number_of_images > 99)
				filename = "0" + std::to_string(number_of_images);
			else
				if (number_of_images > 9)
					filename = "00" + std::to_string(number_of_images);
				else
					filename = "000" + std::to_string(number_of_images);
		}
		else // If the folder is empty
			filename = "0001";
		cv::imwrite("./snapshots/" + filename + ".png", input_image);

		// Show the button capture_video instead of the button snapshot
		capture_video_button.show();

		// Enable the button find_faces 
		find_faces_button.enable();
	}
	
	/*
		This function starts grabbing frames from the camera, 
		clears all the widgets, and disables some buttons
	*/
	void on_capture_video_button_clicked()
	{
		// Tell the thread to unpause itself. This causes should_stop() 
		// to unblock and to let the thread continue.
		start();

		// Hide the button capture_video
		capture_video_button.hide();

		// Clear the labels
		predicted_person_name_label_1.set_text(" ");
		predicted_face_angle.set_text(" ");

		// Hide the face images from previous detection
		detected_face_image_widget.hide();
		predicted_face_image_widget.hide();

		// Disable buttons find_faces, recognize, and add_to_database
		find_faces_button.disable();
		recognize_button.disable();
		add_to_database_button.disable();

		// Show the button snapshot
		snapshot_button.show();
	}

	/*
		This function implements the algorithm of the face pose estimation by its landmarks.
		This algorithm is described in detail here: 
		https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

		@param image_points Facial landmarks.
		The result of executing this function is the angle of rotation of the face 
		about the vertical axis, which is written into the face_angle variable
	*/
	void estimate_pose(std::vector<cv::Point2d> image_points) {
		// Initialize 3D model points
		std::vector<cv::Point3d> model_points;
		model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));          // Nose tip
		model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));     // Chin
		model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));  // Left eye left corner
		model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));   // Right eye right corner
		model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f)); // Left Mouth corner
		model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));  // Right mouth corner

		// Initialize camera internals
		double focal_length = input_image.cols; // Approximate focal length.
		cv::Point2d center = cv::Point2d(input_image.cols / 2, input_image.rows / 2);
		cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
		cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

		// Declare output rotation and translation vectors
		cv::Mat rotation_vector; // Rotation in axis-angle form
		cv::Mat translation_vector;

		// Find a face pose from 3D-2D point correspondences using iterative method
		cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

		//--- TRANSFORM ROTATION AND TRANSLATION VECTORS INTO EULER ANGLES
		cv::Mat rotation_matrix;
		cv::Mat pose_mat;
		cv::Mat euler_angles;
		cv::Rodrigues(rotation_vector, rotation_matrix);
		cv::hconcat(rotation_matrix, translation_vector, pose_mat);
		cv::Mat cameraMatrix;
		cv::Mat rotMatrix;
		cv::Mat	transVect;
		cv::decomposeProjectionMatrix(pose_mat, cameraMatrix, rotMatrix, transVect, cv::noArray(), cv::noArray(), cv::noArray(), euler_angles);

		// Round the calculated angle in such a way that the recognition accuracy is maximum
		// The numbers below was obtained experimentally.
		double calculated_face_angle = euler_angles.at<double>(0, 1);
		if (abs(calculated_face_angle) < 48.0)
			face_angle = 0;
		else if (abs(calculated_face_angle) < 70.0) {
			if (calculated_face_angle > 0)
				face_angle = 60;
			else
				face_angle = -60;
		}
		else {
			if (calculated_face_angle > 0)
				face_angle = 80;
			else
				face_angle = -80;
		}
	}

	/*
		This function detects faces in the image and facial landmarks of the first 
		detected face, and then displays the detection results in the input image widget.
	*/
	void on_find_faces_button_clicked()
	{
		// Delete old detection data
		detected_face_images.clear();
		bbox_text_coordinates.clear();
		
		// Run the face detector on the image. It will return a list of bounding boxes
		// around all the faces in the image.
		std::vector<face_box> dets;
		face_detector.Detect(input_image, dets);

		// If no face is detected
		if (dets.size() == 0) {
			detected_face_label.set_text("No faces found");
			dlib::matrix<dlib::rgb_pixel> face_is_not_found;
			load_image(face_is_not_found, "./face_database/face_is_not_found.bmp");
			detected_face_image_widget.set_image(face_is_not_found);
			// Disable the button add_to_database
			add_to_database_button.disable();
		}
		else { // If a face is detected
			// Initialize normal face landmarks for a 112x112 face image 
			cv::Mat src(5, 2, CV_32FC1, norm_face);

			//--- GET ALIGNED FACE, FACIAL LANDMARKS, AND BOUNDING BOX COORDINATES FOR EACH DETECTED FACE
			for (int i = 0; i < dets.size(); ++i)
			{
				NDArray::WaitAll();
				face_box detection = dets[i];
				float face_landmarks[5][2] =
				{ { detection.landmark.x[0] , detection.landmark.y[0] },
				{ detection.landmark.x[1] , detection.landmark.y[1] },
				{ detection.landmark.x[2] , detection.landmark.y[2] },
				{ detection.landmark.x[3] , detection.landmark.y[3] },
				{ detection.landmark.x[4] , detection.landmark.y[4] } };
				cv::Mat dst(5, 2, CV_32FC1, face_landmarks);

				// Do similar transformation according to a normal face
				cv::Mat m = similarTransform(dst, src);
				cv::Mat aligned_face(112, 112, CV_32FC3);
				cv::Size size(112, 112);

				// Get aligned face with transformed matrix and resize to 112x112
				cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));
				cv::warpAffine(input_image, aligned_face, transfer, size, 1, 0, 0);

				// Draw bounding box of the detected face
				cv::Point bbox_top_left_coord(detection.x0, detection.y0);
				cv::Point bbox_lower_right_coord(detection.x1, detection.y1);
				cv::Point bbox_text_coord(detection.x0, detection.y0 - 10);
				cv::rectangle(input_image, bbox_top_left_coord, bbox_lower_right_coord, cv::Scalar(0, 255, 0), 2);

				// Rearrange facial landmarks so that they correspond to the points of the 3D model
				std::vector<cv::Point2d> landmarks{ cv::Point2d(int(round(dets[i].landmark.x[2])), int(round(dets[i].landmark.y[2]))),    // Nose tip
													cv::Point2d(int(round((dets[i].landmark.x[3] + dets[i].landmark.x[4]) / 2)),          // Chin
													int(round((dets[i].landmark.y[3] - dets[i].landmark.y[2]) + dets[i].landmark.y[3]))),
													cv::Point2d(int(round(dets[i].landmark.x[0])), int(round(dets[i].landmark.y[0]))),    // Left eye left corner
													cv::Point2d(int(round(dets[i].landmark.x[1])), int(round(dets[i].landmark.y[1]))),    // Right eye right corner
													cv::Point2d(int(round(dets[i].landmark.x[3])), int(round(dets[i].landmark.y[3]))),    // Left Mouth corner
													cv::Point2d(int(round(dets[i].landmark.x[4])), int(round(dets[i].landmark.y[4]))) };  // Right mouth corner

				// Show the detected facial landmarks
				for (int i = 0; i < landmarks.size(); i++)
				{
					circle(input_image, landmarks[i], 3, cv::Scalar(0, 0, 255), -1);
				}

				// Save images of aligned faces, facial landmarks, bounding box coordinates
				detected_face_images.push_back(std::move(aligned_face));
				facial_landmarks.push_back(landmarks);
				bbox_text_coordinates.push_back(bbox_text_coord);
			}

			// Run the pose estimation algorithm on the first detected face
			estimate_pose(facial_landmarks[0]);
			
			// Show the estimated angle of the first detected face
			if (face_angle > 0)
				predicted_face_angle.set_text("Face angle: " + std::to_string(face_angle) + "°");
			else
				predicted_face_angle.set_text("Face angle: -" + std::to_string(face_angle) + "°");

			// Turn OpenCV's Mat into variable dlib can deal with
			dlib::cv_image<dlib::bgr_pixel> temp(detected_face_images.front());

			// Show the detected face image in detected_face_image_widget
			detected_face_image_widget.set_image(temp);
			detected_face_image_widget.show();
			detected_face_label.set_text("Detected face:");

			// Show the input image in the main window
			display_input_image();
			
			//--- SAVE THE FIRST DETECTED FACE ON DISK
			dlib::create_directory("detected_faces");
			dlib::directory detected_faces_dir("./detected_faces/");
			std::vector<dlib::file> files = detected_faces_dir.get_files();
			std::string filename;
			if (files.size() > 0)
			{ // If the folder is not empty
				sort(files.begin(), files.end());
				filename = files[files.size() - 1].name();
				filename.erase(filename.length() - 4);
				int number_of_images = stoi(filename) + 1;
				if (number_of_images > 99)
					filename = "0" + std::to_string(number_of_images);
				else
					if (number_of_images > 9)
						filename = "00" + std::to_string(number_of_images);
					else
						filename = "000" + std::to_string(number_of_images);
			}
			else // If the folder is empty
				filename = "0001";
			cv::imwrite(detected_faces_dir.full_name() + "/" + filename + ".png", detected_face_images.front());

			// Enable buttons add_to_database and recognize_button
			add_to_database_button.enable();
			recognize_button.enable();
		}
	}

	/*
		This function implements the face recognition algorithm, as a result of which 
		the names of the persons whose faces has been detected are predicted and 
		displayed in the input image widget
	*/
	void on_recognize_button_clicked()
	{
		//--- APPLY THE FACE RECOGNITION ALGORITHM TO EVERY DETECTED FACE
		for (int i = 0; i < detected_face_images.size(); ++i)
		{
			std::string predicted_name;
			int min_index(0);
			double min_distance(128);

			/*
				Extract a 128 element vector from the facial image using the CNN. The vectors extracted from 
				facial images of the same person have close values so the distance between them is small. 
				If we calculate the distance between the vectors extracted from the facial images of the 
				different persons, then it will be large. We use these vectors to identify if a pair of 
				the facial images are from the same person or from different people.
			*/
			cv::Mat input_feature_vector = feature_extractor.extractFeature(detected_face_images[i]);
			cv::Mat feature_vector;

			// Run the pose estimation algorithm on a detected face
			estimate_pose(facial_landmarks[i]);

			// FIND A FEATURE VECTOR FROM THE FACE DATABASE WITH A MINIMUM DISTANCE TO THE INPUT FEATURE VECTOR
			for (int j = 0; j < face_database.size(); j++)
			{ 
				/*
					Retrieve the feature vector that has been extracted from the face 
					with the same angle of rotation as the input face
				*/
				switch (face_angle) {
				case 0:
					feature_vector = face_database[j].feature_vector_0;
					break;
				case 60:
					feature_vector = face_database[j].feature_vector_45;
					break;
				case -60:
					feature_vector = face_database[j].feature_vector_m45;
					break;
				case 80:
					feature_vector = face_database[j].feature_vector_90;
					break;
				case -80:
					feature_vector = face_database[j].feature_vector_m90;
					break;
				default:
					break;
				}

				// Calculate the vector difference
				cv::Mat difference = (input_feature_vector - feature_vector);

				// Calculate the square of the difference
				cv::Mat difference_squared = difference.mul(difference);

				// Calculate the distance between the vectors
				double distance = cv::sum(difference_squared)[0];

				// Check if the calculated distance is the minimum
				if (distance < min_distance)
				{
					min_distance = distance;
					min_index = j;
				}
			}

			/*
				Check if the detected face does not belong to any person from the face database.
				This can be done by comparing the minimum distance with a threshold that has been 
				selected experimentally
			*/
			if (min_distance < 1.05)
			{ // The person is in the face database
				predicted_name = face_database[min_index].person_name;

				// Show the predicted person's name
				cv::putText(input_image, predicted_name, bbox_text_coordinates[i], cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				
				// Show the first recognition result in the widgets
				if (i == 0)
				{
					predicted_person_name_label_1.set_text(predicted_name);
					dlib::cv_image<dlib::bgr_pixel> temp(face_database[min_index].face_image);
					predicted_face_image_widget.set_image(temp);
				}
			}
			else
			{ // The person is not in the face database
				predicted_name = "Unknown";

				// Show the predicted person's name
				cv::putText(input_image, predicted_name, bbox_text_coordinates[i], cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				
				// Show the first recognition result in the widgets
				if (i == 0)
				{
					predicted_person_name_label_1.set_text("Person is not recognized");
					dlib::matrix<dlib::rgb_pixel> unknown_face_image;
					dlib::load_image(unknown_face_image, "./face_database/unknown_face.bmp");
					predicted_face_image_widget.set_image(unknown_face_image);
				}
			}
		}

		predicted_face_image_widget.show();
		display_input_image();
	}

	// Show the input image in the main window
	void display_input_image() {
		// Turn OpenCV's Mat into variable dlib can deal with
		dlib::cv_image<dlib::bgr_pixel> temp(input_image);
		input_image_widget.set_image(temp);
	}

	/*
		This function adds the first detected face to the face database and saves all data that
		necessary for recognition on disk so that it can be loaded the next time the program is started
	*/
	void add_to_face_database()
	{
		// Get the name specified by the user in add window
		std::string person_name = add_window1.text_field1.text();

		add_window1.hide();
		dlib::create_directory("./face_database/" + person_name);
		dlib::directory face_database_dir("./face_database/" + person_name);

		// Extract feature vector from the first detected face image
		cv::Mat feature_vector = feature_extractor.extractFeature(detected_face_images.front());
		cv::FileStorage feature_vectors_storage(face_database_dir.full_name() + "/feature_vectors.xml", cv::FileStorage::WRITE);

		//--- ADD THE FIRST DETECTED FACE TO THE FACE DATABASE
		std::vector<dlib::file> files = face_database_dir.get_files();
		std::string filename;
		int number_of_images(1);
		if (files.size() > 1)
		{ // If the folder is not empty
			sort(files.begin(), files.end());
			filename = files[files.size() - 2].name();
			filename.erase(filename.length() - 4);
			number_of_images = stoi(filename) + 1;
			filename = std::to_string(number_of_images);
			if (number_of_images > 99)
				filename = "0" + std::to_string(number_of_images);
			else
				if (number_of_images > 9)
					filename = "00" + std::to_string(number_of_images);
				else
					filename = "000" + std::to_string(number_of_images);

			// Find the index of the database entry with the name specified by the user
			unsigned long face_index;
			for (unsigned long i = 0; i < face_database.size(); i++)
			{
				if (face_database[i].person_name == person_name) {
					face_index = i;
				}
			}

			// Save the feature vector taking into account the calculated angle of rotation of the face
			switch (face_angle) {
			case 0:
				face_database[face_index].feature_vector_0 = feature_vector;
				break;
			case 60:
				face_database[face_index].feature_vector_45 = feature_vector;
				break;
			case -60:
				face_database[face_index].feature_vector_m45 = feature_vector;
				break;
			case 80:
				face_database[face_index].feature_vector_90 = feature_vector;
				break;
			case -80:
				face_database[face_index].feature_vector_m90 = feature_vector;
				break;
			default:
				break;
			}
			feature_vectors_storage << "feature_vector_0" << face_database[face_index].feature_vector_0;
			feature_vectors_storage << "feature_vector_45" << face_database[face_index].feature_vector_45;
			feature_vectors_storage << "feature_vector_90" << face_database[face_index].feature_vector_90;
			feature_vectors_storage << "feature_vector_m45" << face_database[face_index].feature_vector_m45;
			feature_vectors_storage << "feature_vector_m90" << face_database[face_index].feature_vector_m90;
		}
		else
		{ // If the folder is empty
			filename = "0001";
			face_database.push_back(Face_data());
			face_database.back().person_name = person_name;
			face_database.back().face_image = detected_face_images.front();
			face_database.back().feature_vector_0 = feature_vector;
			face_database.back().feature_vector_45 = feature_vector;
			face_database.back().feature_vector_90 = feature_vector;
			face_database.back().feature_vector_m45 = feature_vector;
			face_database.back().feature_vector_m90 = feature_vector;
			feature_vectors_storage << "feature_vector_0" << feature_vector;
			feature_vectors_storage << "feature_vector_45" << feature_vector;
			feature_vectors_storage << "feature_vector_90" << feature_vector;
			feature_vectors_storage << "feature_vector_m45" << feature_vector;
			feature_vectors_storage << "feature_vector_m90" << feature_vector;
		}

		// Save the detected facial image of the person
		cv::imwrite(face_database_dir.full_name() + "/" + filename + ".png", detected_face_images.front());
		feature_vectors_storage.release();
	}

	/*
		This function opens a window where the user can select a file to open.
		After selecting a file, the set_image(image_path) function takes the path to the 
		selected file as an argument
	*/
	void open_file() {
		open_existing_file_box(*this, &main_window::set_image);
	}

	// This function opens an image using the given image path
	void set_image(const std::string& image_path) {
		input_image = cv::imread(image_path);
		display_input_image();
		capture_video_button.show();
		find_faces_button.enable();
	}

	// This function show the add window
	void show_add_window() {
		dlib::cv_image<dlib::bgr_pixel> temp(detected_face_images.front());
		add_window1.input_image_widget.set_image(temp);
		add_window1.list_box1.load(add_window1.person_names);
		add_window1.show();
	}

	/*
		This function opens a window where the user can select a directory to save the input image.
		After selecting a directory and a file name, the save_image(image_path) function takes 
		the selected path as an argument
	*/
	void open_save_file_box() {
		save_file_box(*this, &main_window::save_image);
	}

	// This function saves an image using the given image path
	void save_image(const std::string& image_path) {
		cv::imwrite(image_path + ".png", input_image);
	}

	// This function opens the message box
	void show_about()
	{
		dlib::message_box("About", "This program is written by Akhmetov Daryn");
	}

	// This object represents one record in the face database
	struct Face_data {
		std::string person_name;

		/*
			For each person in the face database, we store 5 feature vectors extracted from 
			the images of faces that differ from each other by the angle of rotation around 
			the vertical axis
		*/
		cv::Mat feature_vector_90;
		cv::Mat feature_vector_45;
		cv::Mat feature_vector_0;
		cv::Mat feature_vector_m45;
		cv::Mat feature_vector_m90;

		cv::Mat face_image;
	};

	/*
		This function runs concurrently with other code. We have to use a thread, because clicking on 
		the button does not interrupt the code execution. This thread updates the input image from the 
		camera with a limit of 30 frames per second.
	*/
	void thread()
	{ 
		auto start = std::chrono::high_resolution_clock::now();
		// This loop runs until we call the stop() function
		while (should_stop() == false)
		{
			// Grab a frame from the camera
			cap.read(input_image);
			
			// Update the input image in the main window
			display_input_image();

			//--- ENSURE THAT THE FRAME REFRESH RATE DOES NOT EXCEED 30 FPS
			auto now = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> elapsed = now - start;
			std::cout << elapsed.count() << std::endl;
			if (elapsed.count() < 33.3)
			{
				dlib::sleep(33.3 - elapsed.count());
			}
			start = std::chrono::high_resolution_clock::now();
		}
	}

	// Declare all main window widgets
	add_window add_window1;
	dlib::label input_image_label;
	dlib::label detected_face_label;
	dlib::label predicted_face_label;
	dlib::label predicted_person_name_label_1;
	dlib::label predicted_face_angle;
	dlib::button snapshot_button;
	dlib::button capture_video_button;
	dlib::button find_faces_button;
	dlib::button recognize_button;
	dlib::button add_to_database_button;
	dlib::image_widget input_image_widget;
	dlib::image_widget detected_face_image_widget;
	dlib::image_widget predicted_face_image_widget;
	dlib::menu_bar menu_bar1;
	dlib::bdf_font arial_36;
	dlib::bdf_font arial_32;
	std::shared_ptr<dlib::font> parial_36;
	std::shared_ptr<dlib::font> parial_32;

	// Declare objects required for face detection and face recognition
	MxNetMtcnn face_detector;
	Mxnet_extract feature_extractor;

	// Declare objects required for grabbing frames from the camera
	cv::VideoCapture cap;
	cv::Mat input_image;

	// This variable stores the estimated face rotation angle
	int face_angle;

	// This vector stores biometric information of all people added to the face database
	std::vector<Face_data> face_database;
	
	// These vectors store the results of face detection
	std::vector<cv::Mat> detected_face_images;
	std::vector<cv::Point> bbox_text_coordinates;
	std::vector<std::vector<cv::Point2d>> facial_landmarks;

};

int main()
{
	// Create the main window
	main_window my_window;

	// Wait until the user closes the window before we let the program terminate
	my_window.wait_until_closed();

	return 0;
}

// Normally, if you built this application on MS Windows in Visual Studio you
// would see a black console window pop up when you ran it.  The following
// #pragma directives tell Visual Studio to not include a console window along
// with your application.  However, if you prefer to have the console pop up as
// well then simply remove these #pragma statements.
#ifdef _MSC_VER
#   pragma comment( linker, "/entry:mainCRTStartup" )
#   pragma comment( linker, "/SUBSYSTEM:WINDOWS" )
#endif
