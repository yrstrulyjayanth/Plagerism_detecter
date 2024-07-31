package edu.northeastern.cs5500.service;

import edu.northeastern.cs5500.model.StudentHomeWork;
import edu.northeastern.cs5500.repository.StudentHomeWorkRepository;
import org.apache.commons.io.IOUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.runners.MockitoJUnitRunner;
import org.springframework.core.env.Environment;
import org.springframework.mock.web.MockMultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.mockito.Matchers.any;
import static org.mockito.Matchers.isA;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.when;

/**
 * @author Praveen Singh
 */
@RunWith(MockitoJUnitRunner.class)
public class UploadAssignmentServiceTest {

    @Mock
    private StudentHomeWorkRepository studentHomeWorkRepository;

    @Mock
    private Environment env;

    @InjectMocks
    private UploadAssignmentService uploadAssignmentService;

    @Test
    public void uploadAssignmentShouldWorkASExpected() throws IOException, URISyntaxException {
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(Objects.requireNonNull(classLoader.getResource("submission1.py").toURI().getPath()));
        FileInputStream inputFile = new FileInputStream(file);
        MockMultipartFile multipartFile = new MockMultipartFile("file", "submission1.py",
                "multipart/form-data", inputFile);

        when(env.getProperty("fileupload.paths.uploadedFiles")).thenReturn("./uploads");
        when(studentHomeWorkRepository.save(any(StudentHomeWork.class))).thenReturn(new StudentHomeWork());

        uploadAssignmentService.uploadAssignment(multipartFile, 1, 1, 1);

    }


    @Test
    public void uploadAssignmentShouldUpdateWhenHomeWorkAlreadySubmitted() throws IOException, URISyntaxException {
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(Objects.requireNonNull(classLoader.getResource("submission2.py").toURI().getPath()));
        FileInputStream inputFile = new FileInputStream(file);
        MockMultipartFile multipartFile = new MockMultipartFile("file", "submission2.py",
                "multipart/form-data", inputFile);

        List<StudentHomeWork> studentHomeWorks = new ArrayList<>();
        studentHomeWorks.add(new StudentHomeWork().withUserId(1));

        when(env.getProperty("fileupload.paths.uploadedFiles")).thenReturn("./uploads");
        when(studentHomeWorkRepository.findByUserIdAndCourseIdAndHomeWorkId(1,1,1))
                .thenReturn(studentHomeWorks);

        uploadAssignmentService.uploadAssignment(multipartFile, 1, 1, 1);

    }

    @Test(expected = NullPointerException.class)
    public void uploadAssignmentShouldThrowException() throws IOException {
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(Objects.requireNonNull(classLoader.getResource("test.py")).getFile());
        FileInputStream inputFile = new FileInputStream(file);
        MockMultipartFile multipartFile = new MockMultipartFile("file", "submission1.py",
                "multipart/form-data", inputFile);
        when(env.getProperty("fileupload.paths.uploadedFiles")).thenReturn("./uploads");
        when(studentHomeWorkRepository.save(any(StudentHomeWork.class))).thenReturn(new StudentHomeWork());

        uploadAssignmentService.uploadAssignment(multipartFile, 1, 1, 1);

    }

    @Test
    public void uploadAssignmentUsingByte() throws IOException {

        when(env.getProperty("fileupload.paths.uploadedFiles")).thenReturn("./uploads");
        when(studentHomeWorkRepository.save(any(StudentHomeWork.class))).thenReturn(new StudentHomeWork());

        uploadAssignmentService.uploadAssignment(IOUtils.toByteArray("praveen"), "praveen", 1, 1, 1);
    }

}
