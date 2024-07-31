package edu.northeastern.cs5500.service;

import edu.northeastern.cs5500.model.User;
import edu.northeastern.cs5500.repository.UserRepository;
import edu.northeastern.cs5500.response.RegisterRequestJSON;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.runners.MockitoJUnitRunner;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;
import static org.mockito.Mockito.*;

/**
 * @author Praveen Singh
 */
@RunWith(MockitoJUnitRunner.class)
public class UserServiceTest{

    @Mock
    private UserRepository userRepository;

    @Mock
    private EmailService emailService;

    @InjectMocks
    private UserService userService;

    @Test
    public void createUserShouldReturnAValidUserObject(){
        RegisterRequestJSON registerRequestJSON = new RegisterRequestJSON()
                .withUserName("praveen")
                .withPassword("singh")
                .withFirstName("a")
                .withLastName("b")
                .withEmail("team207@gmail.com");

        User user = userService.createUserObject(registerRequestJSON);

        assertEquals(registerRequestJSON.getUserName(), user.getUserName());
        assertEquals(registerRequestJSON.getPassword(), user.getPassword());
        assertEquals(registerRequestJSON.getEmail(), user.getEmail());
        assertEquals(registerRequestJSON.getLastName(), user.getLastName());
        assertEquals(registerRequestJSON.getFirstName(), user.getFirstName());
        assertEquals(Integer.valueOf(1), user.getRole());
    }

    @Test
    public void findUserByUserNameOrIdShouldReturnEmptyListIfBothPassedAsNull(){
        List<User> result = userService.findUserByUserIdOrUserName(null, null);
        assertTrue(result.isEmpty());
    }

    @Test
    public void findUserByUserNameOrIdShouldReturnUserWhenIdIsPassed(){
        User user = new User().withId(1).withUserName("praveen");
        when(userRepository.findById(1)).thenReturn(user);

        List<User> result = userService.findUserByUserIdOrUserName("1", null);

        verify(userRepository, times(1)).findById(1);
        assertFalse(result.isEmpty());
        assertEquals(user, result.get(0));
    }

    @Test
    public void findUserByUserNameOrIdShouldReturnUserWhenUserNameIsPassed(){
        String userName = "praveen";
        User user = new User().withId(1).withUserName("praveen");
        List<User> userList = new ArrayList<>();
        userList.add(user);
        when(userRepository.findByUserNameLike("%" + userName + "%")).thenReturn(userList);

        List<User> result = userService.findUserByUserIdOrUserName(null, userName);

        verify(userRepository, times(1)).findByUserNameLike("%" + userName + "%");
        assertFalse(result.isEmpty());
        assertEquals(user, result.get(0));
    }

    @Test
    public void checkIfUserExistsShouldReturnTrue(){
        when(userRepository.existsByUserName("praveen")).thenReturn(true);

        boolean res = userRepository.existsByUserName("praveen");

        verify(userRepository, times(1)).existsByUserName("praveen");
        assertTrue(res);
    }

    @Test
    public void checkIfUserExistsShouldReturnFalse(){
        when(userRepository.existsByUserName("praveen")).thenReturn(false);

        boolean res = userService.checkIfUserExistsByUserName("praveen");

        verify(userRepository, times(1)).existsByUserName("praveen");
        assertFalse(res);
    }

    @Test
    public void addUserAndSendEmailShouldWorkAsExpected(){
        RegisterRequestJSON registerRequestJSON = new RegisterRequestJSON()
                .withUserName("praveen")
                .withPassword("singh")
                .withFirstName("praveen")
                .withLastName("singh")
                .withEmail("team207@gmail.com");
        User user = userService.createUserObject(registerRequestJSON);

        when(userRepository.save(user)).thenReturn(user);
        doNothing().when(emailService).sendEmail(registerRequestJSON.getUserName(),
                registerRequestJSON.getEmail(), null);

        userService.addUserAndSendEmail(user);

        verify(userRepository, times(1)).save(user);
        verify(emailService, times(1)).sendEmail(registerRequestJSON.getUserName(),
                registerRequestJSON.getEmail(), null);
    }

    @Test
    public void findByUserNameAndPasswordShouldReturnValidResults(){
        String userName = "praveen";
        String password = "singh";
        User user = new User().withUserName(userName).withPassword(password);
        List<User> users = new ArrayList<>();
        users.add(user);
        when(userRepository.findByUserNameAndPassword(userName, password)).thenReturn(users);

        List<User> userResult = userService.findUserByUserNameAndPassword(userName, password);

        assertNotNull(userResult.get(0));
        assertEquals(userName, userResult.get(0).getUserName());
        assertEquals(password, userResult.get(0).getPassword());

    }

    @Test
    public void testUpdateUser(){
        User user = new User().withId(1).withPassword("praveen").withUserName("singh")
                .withEmail("pr@gmail.com").withFirstName("pk").withLastName("sin");
        when(userRepository.save(user)).thenReturn(null);
        userService.updateUser(user);
    }

}
